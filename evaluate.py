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
import os
import pickle
import time

import torch
from tensorboardX import SummaryWriter

import pips_utils.basic
import pips_utils.improc
import pips_utils.misc
import pips_utils.samp
import pips_utils.test
from datasets.factory import DataloaderFactory
from evaluation_model.factory import EvaluationModelFactory
from evaluation_model.model import EvaluationModel
from pips_utils.figures import compute_summary_df, make_figures
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
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--no_visualisations', action="store_true", default=False)

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

    writer_t = SummaryWriter(os.path.join(args.log_dir, experiment_name, "t"), max_queue=10, flush_secs=60)
    model = EvaluationModelFactory.get_model(args.modeltype, args.checkpoint_path, args.device,
                                             args.pips_stride, args.pips_window)
    dataloader = DataloaderFactory.get_dataloader(args.dataset_type, args.dataset_location, args.subset,
                                                  args.query_mode, args.pips_window, args.flt_use_augs,
                                                  args.flt_crop_size, args.n_points, args.batch_size,
                                                  args.shuffle, args.dataloader_workers)

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
            packed_results = evaluate_batch(args.modeltype, model, batch, args.dataset_type, args.device, sw_t,
                                            args.no_visualisations)
            unpacked_results = EvaluationModel.unpack_results(packed_results, batch_idx)
            results_list += unpacked_results
            summary_df = compute_summary_df(unpacked_results)
            selected_metrics = ["ade_visible", "occlusion_accuracy", "average_jaccard", "average_pts_within_thresh"]
            print(summary_df[selected_metrics].to_markdown())

        iter_time = time.time() - iter_start_time
        print(f'{experiment_name} step={batch_idx:06d} readtime={read_time:>2.2f} itertime={iter_time:>2.2f}')
        read_start_time = time.time()
    writer_t.close()

    metadata = {
        "name": args.modeltype,
        "model": args.modeltype,
        "dataset": f"{args.dataset_type}_{args.subset}",
        "query_mode": args.query_mode,
    }
    save_results(results_list, args.log_dir, experiment_name, args.mostly_visible_threshold, metadata)


def evaluate_batch(modeltype: str, model: EvaluationModel, batch, dataset: str, device,
                   summary_writer, no_visualisations: bool):
    rgbs, query_points, trajectories_gt, visibilities_gt = DataloaderFactory.unpack_batch(batch, dataset, modeltype,
                                                                                          device)
    results = model.evaluate_batch(trajectories_gt, visibilities_gt, rgbs, query_points, summary_writer)
    if not no_visualisations and summary_writer is not None and summary_writer.save_this:
        log_batch_visualisations(summary_writer, rgbs, results["trajectories_gt"], results["trajectories_pred"])
    return results


def log_batch_visualisations(summary_writer: pips_utils.improc.Summ_writer, rgbs, trajectories_gt, trajectories_pred):
    # Plot ground truth trajectories on input RGBs
    rgbs = pips_utils.improc.preprocess_color(rgbs)
    summary_writer.summ_traj2ds_on_rgbs(
        name='inputs_0/orig_trajs_on_rgbs',
        trajs=trajectories_gt,
        rgbs=rgbs,
        cmap='winter',
        linewidth=2,
    )

    # Plot predicted trajectories on output RGBs
    summary_writer.summ_traj2ds_on_rgbs(
        name='outputs/trajs_on_rgbs',
        trajs=trajectories_pred[0:1],
        rgbs=rgbs[0:1],
        cmap='spring',
        linewidth=2,
    )

    # Plot ground truth trajectories on input RGBs with only the trajectories shown
    gt_rgb = summary_writer.summ_traj2ds_on_rgb(
        name='inputs_0_all/single_trajs_on_rgb',
        trajs=trajectories_gt[0:1],
        rgb=torch.mean(rgbs[0:1], dim=1),
        cmap='winter',
        only_return=True,
        linewidth=2,
    )
    gt_rgb = pips_utils.improc.preprocess_color(gt_rgb)

    # Plot ground truth trajectories on a black RGB with only the trajectories shown
    gt_black = summary_writer.summ_traj2ds_on_rgb(
        name='inputs_0_all/single_trajs_on_rgb',
        trajs=trajectories_gt[0:1],
        rgb=torch.ones_like(rgbs[0:1, 0]) * -0.5,
        cmap='winter',
        only_return=True,
        linewidth=2,
    )
    gt_black = pips_utils.improc.preprocess_color(gt_black)

    # Plot predicted trajectories on input RGBs with ground truth trajectories
    summary_writer.summ_traj2ds_on_rgb(
        name='outputs/single_trajs_on_gt_rgb',
        trajs=trajectories_pred[0:1],
        rgb=gt_rgb[0:1],
        cmap='spring',
        linewidth=2,
    )

    # Plot predicted trajectories on black RGB with ground truth trajectories
    summary_writer.summ_traj2ds_on_rgb(
        name='outputs/single_trajs_on_gt_black',
        trajs=trajectories_pred[0:1],
        rgb=gt_black[0:1],
        cmap='spring',
        linewidth=2,
    )


def save_results(results_list, log_dir, model_name, mostly_visible_threshold, metadata):
    # TODO: Save the results to a JSON file instead of a pickle file because it's more human-readable
    results_list_pkl_path = os.path.join(log_dir, model_name, "results_list.pkl")
    with open(results_list_pkl_path, "wb") as f:
        print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
        pickle.dump(results_list, f)
    results_df_path = os.path.join(log_dir, model_name, "results_df.csv")
    results_df = compute_summary_df(results_list)
    results_df.to_csv(results_df_path)
    print(f"\nResults summary dataframe saved to:\n{results_df_path}\n")
    figures_dir = os.path.join(log_dir, model_name, "figures")
    ensure_dir(figures_dir)
    for k, v in metadata.items():
        results_df[k] = v
    make_figures(results_df, figures_dir, mostly_visible_threshold)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"   {arg}: {getattr(args, arg)}")
    evaluate(args)
