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
import os
import pickle
import random
import time
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from fire import Fire
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import pips_utils.basic
import pips_utils.improc
import pips_utils.misc
import pips_utils.samp
import pips_utils.test
import saverloader
from datasets.flyingthings import FlyingThingsDataset
from datasets.tapvid import TAPVidIterator
from nets.pips import Pips
from nets.raftnet import Raftnet
from pips_utils.figures import compute_summary_df, make_figures
from pips_utils.util import ensure_dir, get_str_formatted_time

device = 'cuda'
random.seed(125)
np.random.seed(125)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
        dataset_type: str,
        dataset_location: str,
        subset: str = 'all',
        shuffle: bool = False,
        dataloader_workers: int = 20,
        seed: int = 72,

        modeltype: str = 'pips',
        init_dir: str = 'reference_model',
        stride: int = 4,

        B: int = 1,
        S: int = 8,
        N: Union[int, None] = None,
        mostly_visible_threshold: int = 4,

        exp_name: str = "test",
        log_freq: int = 50,
        log_dir: str = 'logs',

        # FlyingThings++ specific
        crop_size: Tuple[int] = (384, 512),  # the raw data is 540,960
        use_augs: bool = False,

        # TAP-Vid specific
        query_mode: str = "strided",
):
    seed_all(seed)

    assert (modeltype == 'pips' or modeltype == 'raft' or modeltype == 'dino')

    model_name = f"{B:d}_{S:d}_{N}_{modeltype:s}"
    if use_augs:
        model_name += "_A"
    if dataset_type == "tapvid":
        model_name += f"_{query_mode}"
    model_name += f"_{exp_name:s}"
    model_date = get_str_formatted_time()
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(os.path.join(log_dir, model_name, "t"), max_queue=10, flush_secs=60)

    if dataset_type == "flyingthings++":
        dataset = FlyingThingsDataset(
            dataset_location=dataset_location,
            dset='TEST', subset=subset,
            use_augs=use_augs,
            N=N, S=S,
            crop_size=crop_size,
        )
        test_dataloader = DataLoader(
            dataset,
            batch_size=B,
            shuffle=shuffle,
            num_workers=dataloader_workers,
            worker_init_fn=worker_seed_init_fn,
            drop_last=True,
        )

    elif dataset_type == "tapvid":
        test_dataloader = TAPVidIterator(dataset_location, subset, query_mode)

    elif dataset_type == "crohd":
        raise NotImplementedError()

    else:
        raise ValueError(f"Invalid dataset type given: `{dataset_type}`")

    if modeltype == 'pips':
        model = Pips(S=S, stride=stride).cuda()
        _ = saverloader.load(init_dir, model)
        model.eval()
    elif modeltype == 'raft':
        model = Raftnet(ckpt_name='raft_ckpts/raft-things.pth').cuda()
        model.eval()
    elif modeltype == 'dino':
        patch_size = 8
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size).cuda()
        model.eval()
    else:
        raise ValueError(f"Invalid modeltype given: `{modeltype}`")

    results_list = []
    read_start_time = time.time()
    for batch_idx, batch in enumerate(test_dataloader):
        sw_t = pips_utils.improc.Summ_writer(
            writer=writer_t,
            global_step=batch_idx,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq / 2),
            just_gif=True,
        )

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            packed_results = evaluate_batch(modeltype, model, batch, sw_t, dataset_type)
            for b in range(packed_results["trajectories_gt"].shape[0]):
                for n in range(packed_results["trajectories_gt"].shape[2]):
                    result = {
                        "iter": batch_idx,
                        "video_idx": b,
                        "point_idx_in_video": n,
                        "trajectory_gt": packed_results["trajectories_gt"][b, :, n, :],
                        "trajectory_pred": packed_results["trajectories_pred"][b, :, n, :],
                        "visibility_gt": packed_results["visibilities_gt"][b, :, n],
                        "query_point": packed_results["query_points"][b, n, :],
                    }
                    results_list += [result]

        iter_time = time.time() - iter_start_time
        print(
            f'{model_name}'
            f' step={batch_idx:06d}'
            f' readtime={read_time:>2.2f}'
            f' itertime={iter_time:>2.2f}'
        )
        read_start_time = time.time()
    writer_t.close()

    save_results(log_dir, model_name, modeltype, mostly_visible_threshold, results_list)


def worker_seed_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def evaluate_batch(modeltype, model, batch, summary_writer, dataset="flyingthings++"):
    rgbs, query_points, trajectories_gt, visibilities_gt = unpack_batch(batch, dataset, modeltype)
    trajectories_pred = forward_pass(model, modeltype, summary_writer,
                                     rgbs, query_points, trajectories_gt, visibilities_gt)

    batch_size = rgbs.shape[0]
    n_frames = rgbs.shape[1]
    n_points = trajectories_gt.shape[2]
    assert trajectories_pred.shape == (batch_size, n_frames, n_points, 2)

    results = {
        "trajectories_gt": trajectories_gt.detach().clone().cpu(),
        "visibilities_gt": visibilities_gt.detach().clone().cpu(),
        "trajectories_pred": trajectories_pred.detach().clone().cpu(),
        "query_points": query_points.detach().clone().cpu(),
    }

    if summary_writer is not None and summary_writer.save_this:
        log_batch_visualisations(summary_writer, rgbs, results["trajectories_gt"], results["trajectories_pred"])
    return results


def unpack_batch(batch, dataset, modeltype):
    # TODO Refactor: Move to respective dataloaders
    if dataset == "flyingthings++":
        rgbs = batch['rgbs'].cuda().float()
        query_points = None  # TODO
        trajectories_gt = batch['trajs'].cuda().float()
        visibilities_gt = batch['visibles'].cuda().float()

    elif dataset == "crohd":
        rgbs = batch['rgbs'].cuda()
        query_points = None  # TODO
        trajectories_gt = batch['trajs_g'].cuda()
        visibilities_gt = batch['vis_g'].cuda()

        batch_size, n_frames, channels, height, width = rgbs.shape
        batch_size, S1, n_points, D = trajectories_gt.shape

        rgbs_ = rgbs.reshape(batch_size * n_frames, channels, height, width)
        if modeltype == "dino":
            H_, W_ = 512, 768
        else:
            H_, W_ = 768, 1280

        sy = H_ / height
        sx = W_ / width
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        height, width = H_, W_
        rgbs = rgbs_.reshape(batch_size, n_frames, channels, height, width)
        trajectories_gt[:, :, :, 0] *= sx
        trajectories_gt[:, :, :, 1] *= sy

    elif dataset == "tapvid":
        rgbs = batch["rgbs"].cuda()
        query_points = batch["query_points"].cuda()
        trajectories_gt = batch["trajectories"].cuda()
        visibilities_gt = batch["visibilities"].cuda()

    else:
        raise ValueError(f"Invalid dataset given: `{dataset}`")

    batch_size, n_frames, channels, height, width = rgbs.shape
    n_points = trajectories_gt.shape[2]

    assert rgbs.shape == (batch_size, n_frames, channels, height, width)
    assert query_points.shape == (batch_size, n_points, 3)
    assert trajectories_gt.shape == (batch_size, n_frames, n_points, 2)
    assert visibilities_gt.shape == (batch_size, n_frames, n_points)

    return rgbs, query_points, trajectories_gt, visibilities_gt


def forward_pass(model, modeltype, summary_writer, rgbs, query_points, trajectories_gt, visibilities_gt):
    # TODO Refactor: Move to respective models, create a wrapper around each model
    if modeltype == 'pips':
        raise NotImplementedError()  # TODO
        model: Pips = model
        preds, preds_anim, vis_e, stats = model(
            xys=trajectories_gt[:, 0],
            rgbs=rgbs,
            iters=6,
            trajs_g=trajectories_gt,
            vis_g=visibilities_gt,
            sw=summary_writer,
        )
        return preds[-1]

    elif modeltype == 'dino':
        raise NotImplementedError()  # TODO
        return pips_utils.test.get_dino_output(model, rgbs, trajectories_gt, visibilities_gt)

    elif modeltype == 'raft':
        model: Raftnet = model
        prep_rgbs = pips_utils.improc.preprocess_color(rgbs)

        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = trajectories_gt.shape[2]

        flows_forward = []
        flows_backward = []
        for t in range(1, n_frames):
            rgb0 = prep_rgbs[:, t - 1]
            rgb1 = prep_rgbs[:, t]
            flows_forward.append(model(rgb0, rgb1, iters=32)[0])
            flows_backward.append(model(rgb1, rgb0, iters=32)[0])
        flows_forward = torch.stack(flows_forward, dim=1)
        flows_backward = torch.stack(flows_backward, dim=1)
        assert flows_forward.shape == flows_backward.shape == (batch_size, n_frames - 1, 2, height, width)

        coords = []
        for t in range(n_frames):
            if t == 0:
                coord = torch.zeros_like(trajectories_gt[:, 0])
            else:
                prev_coord = coords[t - 1]
                delta = pips_utils.samp.bilinear_sample2d(
                    im=flows_forward[:, t - 1],
                    x=prev_coord[:, :, 0],
                    y=prev_coord[:, :, 1],
                ).permute(0, 2, 1)
                assert delta.shape == (batch_size, n_points, 2), "Forward flow at the discrete points"
                coord = prev_coord + delta

            # Set the ground truth query point location if hte timestep is correct
            query_point_mask = query_points[:, :, 0] == t
            coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

            coords.append(coord)

        for t in range(n_frames - 2, -1, -1):
            coord = coords[t]
            successor_coord = coords[t + 1]

            delta = pips_utils.samp.bilinear_sample2d(
                im=flows_backward[:, t],
                x=successor_coord[:, :, 0],
                y=successor_coord[:, :, 1],
            ).permute(0, 2, 1)
            assert delta.shape == (batch_size, n_points, 2), "Backward flow at the discrete points"

            # Update only the points that are located prior to the query point
            prior_to_query_point_mask = t < query_points[:, :, 0]
            coord = (coord * ~prior_to_query_point_mask.unsqueeze(-1) +
                     (successor_coord + delta) * prior_to_query_point_mask.unsqueeze(-1))
            coords[t] = coord

        return torch.stack(coords, dim=1)

    else:
        raise ValueError(f"Invalid modeltype given: `{modeltype}`")


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


def save_results(log_dir, model_name, modeltype, mostly_visible_threshold, results_list):
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
    results_df["name"] = modeltype
    make_figures(results_df, figures_dir, mostly_visible_threshold)


if __name__ == '__main__':
    Fire(evaluate)
