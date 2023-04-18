"""
Evaluate on FlyingThings++ or TAP-Vid. Supports evaluating PIPs, RAFT, and DINO.

Examples of usage:
```bash
python test_on_flt.py
python test_on_flt.py --modeltype raft
python test_on_flt.py --modeltype dino
python test_on_flt.py --modeltype dino --seed 123
python test_on_flt.py --modeltype dino --seed 123 --N 64
```
"""
import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
from segment_anything import sam_model_registry, SamPredictor
from tensorboardX import SummaryWriter
from tqdm import tqdm

import pips_utils.basic
import pips_utils.improc
import pips_utils.misc
import pips_utils.samp
import pips_utils.test
import wandb
from datasets.factory import DataloaderFactory
from evaluation_model.factory import EvaluationModelFactory
from evaluation_model.model import EvaluationModel
from pips_utils.figures import compute_summary_df, make_figures, compute_summary
from pips_utils.util import ensure_dir, get_str_formatted_time, seed_all


def get_parser():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset_type', type=str, required=True)
    parser.add_argument('--dataset_location', type=str, required=True)
    parser.add_argument('--subset', type=str, default='all')
    parser.add_argument('--dataloader_workers', type=int, default=20)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=72)

    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_points', type=int, default=None)

    # Model
    parser.add_argument('--modeltype', type=str, default='pips')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pips_stride', type=int, default=4)

    # General
    parser.add_argument('--pips_window', type=int, default=8)
    parser.add_argument('--mostly_visible_threshold', type=int, default=4)

    # Logging
    parser.add_argument('--wandb_entity', type=str, default='point-tracking')
    parser.add_argument('--wandb_project', type=str, default='evaluation-debugging')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--no_visualisations', action="store_true", default=False)
    parser.add_argument('--dont_save_raw_results', action="store_true", default=False)
    parser.add_argument('--log_sam_output', action="store_true", default=False)

    # FlyingThings++ specific
    parser.add_argument('--flt_crop_size', type=int, nargs=2, default=(384, 512))
    parser.add_argument('--flt_use_augs', action='store_true', default=False)

    # TAP-Vid specific
    parser.add_argument('--query_mode', type=str, default='strided')

    return parser


def evaluate(args):
    seed_all(args.seed)

    assert (args.modeltype == 'pips' or args.modeltype == 'raft' or args.modeltype == 'dino')

    # TODO save the metadata to a file and read it from there
    if args.experiment_name is None:
        experiment_name = f"{args.batch_size:d}_{args.pips_window:d}_{args.n_points}_{args.modeltype:s}"
        experiment_name += f"_{args.seed}"
        experiment_name += f"_{args.dataset_type}"
        experiment_name += f"_{args.subset}"
        if args.flt_use_augs:
            experiment_name += "_useaugs"
        if args.dataset_type == "tapvid":
            experiment_name += f"_{args.query_mode}"
        experiment_name += f"_{get_str_formatted_time()}"
    else:
        experiment_name = args.experiment_name + f"_{get_str_formatted_time()}"
    print(f"experiment_name={experiment_name}")

    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=experiment_name, config=vars(args),
               sync_tensorboard=True)
    print(f"Wandb run url: {wandb.run.url}")

    output_dir = os.path.join(args.log_dir, experiment_name)
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf8") as f:
        json.dump(vars(args), f, indent=4)

    writer_t = SummaryWriter(os.path.join(output_dir, "t"), max_queue=10, flush_secs=60)
    model = EvaluationModelFactory.get_model(args.modeltype, args.checkpoint_path, args.device,
                                             args.pips_stride, args.pips_window)
    dataloader = DataloaderFactory.get_dataloader(args.dataset_type, args.dataset_location, args.subset,
                                                  args.query_mode, args.pips_window, args.flt_use_augs,
                                                  args.flt_crop_size, args.n_points, args.batch_size,
                                                  args.shuffle, args.dataloader_workers)

    summaries = []
    results_list = []
    read_start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if args.max_iter is not None and batch_idx >= args.max_iter:
            break

        sw_t = pips_utils.improc.Summ_writer(
            writer=writer_t,
            global_step=batch_idx,
            log_freq=args.log_freq,
            fps=5,
            scalar_freq=int(args.log_freq / 2),
            just_gif=True,
        )

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            rgbs, query_points, trajectories_gt, visibilities_gt = DataloaderFactory.unpack_batch(
                batch=batch,
                dataset=args.dataset_type,
                modeltype=args.modeltype,
                device=args.device,
            )
            packed_results = model.evaluate_batch(trajectories_gt, visibilities_gt, rgbs, query_points, sw_t)
            unpacked_results = EvaluationModel.unpack_results(packed_results, batch_idx)
            summaries_batch = [compute_summary(res) for res in unpacked_results]
            summaries += summaries_batch

            summary_df = compute_summary_df(unpacked_results)
            selected_metrics = ["ade_visible", "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy"]
            selected_metrics_shorthand = {
                "ade_visible": "ADE",
                "average_jaccard": "AJ",
                "average_pts_within_thresh": "<D",
                "occlusion_accuracy": "OA",
            }
            print(summary_df[selected_metrics].to_markdown())

            if not args.no_visualisations and sw_t is not None and sw_t.save_this:
                log_batch_visualisations(sw_t, rgbs, query_points, packed_results, unpacked_results, summaries_batch,
                                         summary_df, selected_metrics, selected_metrics_shorthand, args.modeltype,
                                         args.log_sam_output)

            if not args.dont_save_raw_results:
                results_list += unpacked_results

        iter_time = time.time() - iter_start_time
        print(f'{experiment_name} step={batch_idx:06d} readtime={read_time:>2.2f} itertime={iter_time:>2.2f}')
        read_start_time = time.time()
    writer_t.close()

    metadata = {
        "name": args.modeltype,
        "model": args.modeltype,
        "dataset": f"{args.dataset_type}_{args.subset}",
        "query_mode": args.query_mode,
        "checkpoint": args.checkpoint_path,
    }
    save_results(summaries, results_list, output_dir, args.mostly_visible_threshold, metadata)


def log_batch_visualisations(
        summary_writer: pips_utils.improc.Summ_writer,
        rgbs,
        query_points,
        packed_results,
        unpacked_results,
        batch_summaries,
        summary_df,
        selected_metrics,
        key_shorthand,
        modeltype,
        log_sam_output,
):
    trajectories_gt = packed_results["trajectories_gt"]
    trajectories_pred = packed_results["trajectories_pred"]
    visibilities_gt = packed_results["visibilities_gt"]
    visibilities_pred = packed_results["visibilities_pred"]

    batch_size, n_frames, channels, height, width = rgbs.shape
    n_points = trajectories_gt.shape[2]
    assert (batch_size == 1), "Only one batch element logging supported"
    assert (channels == 3)

    # All points
    max_key_len = max([len(key_shorthand[k]) for k in selected_metrics])
    summary_template_str = f"\n{{k:>{max_key_len}s}}: {{v:.4f}}"
    prefix = f"all_points_step-{summary_writer.global_step}/"
    text_annotation = f"{modeltype}"
    text_annotation += f"\nall_step-{summary_writer.global_step}"
    text_annotation += f"\n{rgbs.shape}"
    for k, v in summary_df[selected_metrics].mean().items():
        text_annotation += "\n" + summary_template_str.format(k=key_shorthand[k], v=v)
    plot_trajectories(
        rgbs=rgbs,
        trajectories_gt=trajectories_gt,
        trajectories_pred=trajectories_pred,
        visibilities_gt=visibilities_gt,
        visibilities_pred=visibilities_pred,
        summary_writer=summary_writer,
        prefix=prefix,
        text_annotation=text_annotation,
    )

    # Selected groups of points (standard, hard, easy)
    standard_points = [
        i for i in range(n_points)
        if i < 5 or (i % 10 == 0 and i < 100) or (i % 100 == 0 and i < 1000)
    ]
    hard_points = [
        i for i in range(n_points)
        if batch_summaries[i]["average_pts_within_thresh"] < 30
           or batch_summaries[i]["occlusion_accuracy"] < 20
           or batch_summaries[i]["average_jaccard"] < 20
           or batch_summaries[i]["ade_visible"] > 20
    ]
    easy_points = [
        i for i in range(n_points)
        if batch_summaries[i]["average_pts_within_thresh"] > 95
           and batch_summaries[i]["occlusion_accuracy"] > 70
           and batch_summaries[i]["average_jaccard"] > 70
           and batch_summaries[i]["ade_visible"] < 5
    ]
    hard_points = hard_points[:5]
    easy_points = easy_points[:5]
    for point_category, points in [
        ("standard", standard_points),
        ("hard", hard_points),
        ("easy", easy_points),
    ]:
        for point_idx in points:
            print(f"Logging visualisations for {point_category} point with index: {point_idx}")
            prefix = f"{point_category}_points_step-{summary_writer.global_step}/point-{point_idx}/"
            text_annotation = f"{modeltype}"
            text_annotation += f"\n{point_category}_step-{summary_writer.global_step}_point-{point_idx}"
            text_annotation += f"\n" + " ".join(f"{x:.2f}" for x in query_points[0, point_idx, :].tolist())
            for k in selected_metrics:
                text_annotation += summary_template_str.format(k=key_shorthand[k], v=batch_summaries[point_idx][k])
            plot_trajectories(
                rgbs=rgbs,
                trajectories_gt=trajectories_gt[:, :, [point_idx], :],
                trajectories_pred=trajectories_pred[:, :, [point_idx], :],
                visibilities_gt=visibilities_gt,
                visibilities_pred=visibilities_pred,
                summary_writer=summary_writer,
                prefix=prefix,
                text_annotation=text_annotation,
                add_sam_output=log_sam_output,
            )
    print("Logging done...")


def plot_trajectories(
        rgbs,
        trajectories_gt,
        trajectories_pred,
        visibilities_gt,
        visibilities_pred,
        summary_writer,
        prefix="",
        text_annotation="your ad can be here",
        add_sam_output=False,
):
    rgbs = pips_utils.improc.preprocess_color(rgbs)
    n_frames = rgbs.shape[1]
    text_annotation_per_frame = [f"{i}\n{text_annotation}" for i in range(n_frames)]

    if add_sam_output:
        log_key = f'{prefix}sam_masks'
        log_rgbs_annotated_with_sam_masks_of_trajectory_points(log_key, rgbs, trajectories_gt, trajectories_pred,
                                                               visibilities_gt, visibilities_pred)

    # Plot ground truth trajectories on a black RGB with only the trajectories shown
    gt_black = summary_writer.summ_traj2ds_on_rgb(
        name=f'{prefix}inputs_0_all/single_trajs_on_rgb',
        trajs=trajectories_gt[0:1],
        rgb=torch.ones_like(rgbs[0:1, 0]) * -0.5,
        cmap='winter',
        only_return=True,
        linewidth=1,
        frame_id=text_annotation,
    )
    # Plot predicted trajectories on black RGB with ground truth trajectories
    gt_black = pips_utils.improc.preprocess_color(gt_black)
    gt_and_pred_black = summary_writer.summ_traj2ds_on_rgb(
        name=f'{prefix}outputs/single_trajs_on_gt_black',
        trajs=trajectories_pred[0:1],
        rgb=gt_black[0:1],
        cmap='spring',
        only_return=True,
        linewidth=1,
        frame_id=text_annotation,
    )

    # Plot ground truth trajectories on input RGBs
    gt_traj_rgbs = summary_writer.summ_traj2ds_on_rgbs(
        name=f'{prefix}inputs_0/orig_trajs_on_rgbs',
        trajs=trajectories_gt,
        rgbs=rgbs,
        cmap='winter',
        linewidth=1,
        only_return=True,
        frame_ids=["" for _ in range(n_frames)],
        valids=visibilities_gt,
    )

    # Plot predicted trajectories on output RGBs
    pred_traj_rgbs = summary_writer.summ_traj2ds_on_rgbs(
        name=f'{prefix}outputs/trajs_on_rgbs',
        trajs=trajectories_pred,
        rgbs=rgbs,
        cmap='spring',
        linewidth=1,
        only_return=True,
        frame_ids=["your ad can be here" for _ in range(n_frames)],
        valids=visibilities_pred,
    )

    gt_traj_rgbs = pips_utils.improc.preprocess_color(gt_traj_rgbs)
    pred_traj_rgbs = pips_utils.improc.preprocess_color(pred_traj_rgbs)
    gt_and_pred_black = pips_utils.improc.preprocess_color(gt_and_pred_black)
    rgbs = torch.cat([
        rgbs.to(gt_traj_rgbs.device),
        gt_traj_rgbs,
        pred_traj_rgbs,
        gt_and_pred_black.repeat(1, n_frames, 1, 1, 1),
    ], dim=4)

    # Plot predicted trajectories on top of ground truth trajectories on input RGBs
    gt_traj_rgbs_2 = summary_writer.summ_traj2ds_on_rgbs(
        name=f'{prefix}inputs_0/orig_trajs_on_rgbs',
        trajs=trajectories_gt,
        rgbs=rgbs,
        cmap='winter',
        linewidth=1,
        only_return=True,
        frame_ids=["" for _ in range(n_frames)],
    )
    gt_traj_rgbs_2 = pips_utils.improc.preprocess_color(gt_traj_rgbs_2)
    pred_traj_rgbs_2 = summary_writer.summ_traj2ds_on_rgbs(
        name=f'{prefix}outputs/trajs_on_rgbs',
        trajs=trajectories_pred,
        rgbs=gt_traj_rgbs_2,
        cmap='spring',
        linewidth=1,
        only_return=False,
        frame_ids=text_annotation_per_frame,
    )


def log_rgbs_annotated_with_sam_masks_of_trajectory_points(log_key, rgbs, trajectories_gt, trajectories_pred,
                                                           visibilities_gt, visibilities_pred, max_masks=100):
    n_frames = rgbs.shape[1]
    n_points = trajectories_gt.shape[2]
    assert n_points == 1

    rgbs = rgbs.permute(0, 1, 3, 4, 2)  # (B, T, C, H, W) -> (B, T, H, W, C)
    rgbs = pips_utils.improc.back2color(rgbs)
    rgbs = rgbs.cpu().numpy()

    # TODO Hardcoded checkpoint
    sam_checkpoint = "sam_checkpoints/sam_vit_h_4b8939.pth"
    sam_model = "vit_h"
    sam_predictor = create_sam_predictor(sam_checkpoint, sam_model)

    masks_gt_list = []
    masks_pred_list = []
    for frame_idx in tqdm(range(n_frames)):
        sam_predictor.set_image(rgbs[0, frame_idx])
        point_labels = np.array([1])  # Assume that the points are foreground points (1), not background points (0)

        # TODO Batchify over all interesting points in a video
        point_coords_gt = trajectories_gt[0, frame_idx, 0, :].cpu().numpy().reshape((1, 2))
        masks_gt, scores, logits = sam_predictor.predict(point_coords_gt, point_labels, multimask_output=False)
        masks_gt = masks_gt[:max_masks, :, :]

        point_coords_pred = trajectories_pred[0, frame_idx, 0, :].cpu().numpy().reshape((1, 2))
        masks_pred, scores, logits = sam_predictor.predict(point_coords_pred, point_labels, multimask_output=False)

        if visibilities_gt[0, frame_idx, 0] == 1:
            masks_gt_list.append(masks_gt)
        else:
            masks_gt_list.append([])

        if visibilities_pred[0, frame_idx, 0] == 1:
            masks_pred_list.append(masks_pred)
        else:
            masks_pred_list.append([])

    masked_rgbs = []
    color_palette = [np.random.random((1, 1, 3)) for _ in range(max_masks)]
    for frame_idx in range(n_frames):
        rgb = rgbs[0, frame_idx]
        masks_gt = masks_gt_list[frame_idx]
        masks_pred = masks_pred_list[frame_idx]
        point_gt = trajectories_gt[0, frame_idx, 0, :].cpu().numpy()
        point_pred = trajectories_pred[0, frame_idx, 0, :].cpu().numpy()

        masked_gt_rgb = overlay_masks_on_rgb(rgb / 255, masks_gt, color_palette)
        masked_pred_rgb = overlay_masks_on_rgb(rgb / 255, masks_pred, color_palette)

        masked_gt_rgb = cv2.circle(masked_gt_rgb, (int(point_gt[0]), int(point_gt[1])), 5, (255, 0, 0), -1)
        masked_pred_rgb = cv2.circle(masked_pred_rgb, (int(point_pred[0]), int(point_pred[1])), 5, (255, 0, 0), -1)

        masked_gt_rgb = (masked_gt_rgb * 255).astype(np.uint8)
        masked_pred_rgb = (masked_pred_rgb * 255).astype(np.uint8)

        masked_rgbs += [np.concatenate([rgb, masked_gt_rgb, masked_pred_rgb], axis=1)]

    log_video_to_wandb(log_key, masked_rgbs)


def create_sam_predictor(checkpoint, model) -> SamPredictor:
    sam = sam_model_registry[model](checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def overlay_masks_on_rgb(rgb, masks, color_palette):
    for mask, color in zip(masks, color_palette):
        rgb = overlay_mask_on_rgb(rgb, mask, color)
    return rgb


def overlay_mask_on_rgb(rgb, mask, color):
    overlay = np.ones_like(rgb)
    overlay *= color
    overlay = np.where(mask[..., None] > 0, overlay, rgb)
    overlaid_rgb = cv2.addWeighted(rgb, 0.3, overlay, 0.7, 0)
    return overlaid_rgb


def log_video_to_wandb(log_key, frames):
    frames_4d = np.stack(frames, axis=0)
    frames_4d = frames_4d.transpose((0, 3, 1, 2))
    wandb.log({log_key: wandb.Video(frames_4d, format="gif", fps=4)})


def save_results(summaries, results_list, output_dir, mostly_visible_threshold, metadata):
    # Save summaries as a json file
    summaries_path = os.path.join(output_dir, "summaries.json")
    with open(summaries_path, "w", encoding="utf8") as f:
        json.dump(summaries, f)
    print(f"\nSummaries saved to:\n{summaries_path}\n")

    # Save results summary dataframe as a csv file
    results_df_path = os.path.join(output_dir, "results_df.csv")
    results_df = pd.DataFrame.from_records(summaries)
    results_df.to_csv(results_df_path)
    print(f"\nResults summary dataframe saved to:\n{results_df_path}\n")
    for k, v in metadata.items():
        results_df[k] = v

    # Save results summary dataframe as a wandb artifact
    artifact = wandb.Artifact(name=f"{wandb.run.name}__results_df", type="df", metadata=metadata)
    artifact.add_file(results_df_path, "results_df.csv")
    wandb.log_artifact(artifact)

    # Save results list as a pickle file
    if len(results_list) > 0:
        results_list_pkl_path = os.path.join(output_dir, "results_list.pkl")
        with open(results_list_pkl_path, "wb") as f:
            print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
            pickle.dump(results_list, f)

    # Make figures
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)
    make_figures(results_df, figures_dir, mostly_visible_threshold)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"   {arg}: {getattr(args, arg)}")
    evaluate(args)
