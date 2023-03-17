"""
Evaluate on FlyingThings++. Supports evaluating PIPs, RAFT, and DINO.

Examples of usage:
```bash
python test_on_flt.py
python test_on_flt.py --modeltype raft
python test_on_flt.py --modeltype dino
python test_on_flt.py --modeltype dino --seed 123
python test_on_flt.py --modeltype dino --seed 123 --N 64
```
"""
import datetime
import os
import pickle
import random
import time

import numpy as np
import torch
from fire import Fire
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import pips_utils.basic
import pips_utils.improc
import pips_utils.test
import saverloader
from flyingthingsdataset import FlyingThingsDataset
from nets.pips import Pips
from nets.raftnet import Raftnet
from pips_utils.figures import figure1, compute_summary_df
from pips_utils.util import ensure_dir

device = 'cuda'
random.seed(125)
np.random.seed(125)


def run_for_sample(modeltype, model, d, sw):
    rgbs = d['rgbs'].cuda().float()
    occs = d['occs'].cuda().float()
    masks = d['masks'].cuda().float()
    trajs_gt = d['trajs'].cuda().float()
    vis_gt = d['visibles'].cuda().float()
    valids = d['valids'].cuda().float()

    B, S, C, H, W = rgbs.shape
    _, _, N, D = trajs_gt.shape

    assert D == 2
    assert C == 3

    assert rgbs.shape == (B, S, C, H, W)
    assert occs.shape == (B, S, 1, H, W)
    assert masks.shape == (B, S, 1, H, W)
    assert trajs_gt.shape == (B, S, N, D)
    assert vis_gt.shape == (B, S, N)
    assert valids.shape == (B, S, N)

    assert (torch.sum(valids) == B * S * N)

    # compute per-sequence visibility labels
    good_visibility_mask = (torch.sum(vis_gt, dim=1, keepdim=True) >= 4).float().repeat(1, S, 1)

    if modeltype == 'pips':
        preds, preds_anim, vis_e, stats = model(trajs_gt[:, 0], rgbs, iters=6, trajs_g=trajs_gt, vis_g=vis_gt,
                                                valids=valids, sw=sw)
        trajs_pred = preds[-1]

    elif modeltype == 'dino':
        trajs_pred = pips_utils.test.get_dino_output(model, rgbs, trajs_gt, vis_gt)

    elif modeltype == 'raft':
        prep_rgbs = pips_utils.improc.preprocess_color(rgbs)

        flows_e = []
        for s in range(S - 1):
            rgb0 = prep_rgbs[:, s]
            rgb1 = prep_rgbs[:, s + 1]
            flow, _ = model(rgb0, rgb1, iters=32)
            flows_e.append(flow)
        flows_e = torch.stack(flows_e, dim=1)
        assert flows_e.shape == (B, S - 1, 2, H, W)

        coords = []
        coord0 = trajs_gt[:, 0]
        coords.append(coord0)
        coord = coord0.clone()
        for s in range(S - 1):
            delta = pips_utils.samp.bilinear_sample2d(flows_e[:, s], coord[:, :, 0], coord[:, :, 1]).permute(0, 2, 1)
            assert delta.shape == (B, N, 2), "Forward flow at the discrete points"
            coord = coord + delta
            coords.append(coord)
        trajs_pred = torch.stack(coords, dim=1)

    else:
        raise ValueError(f"Invalid modeltype given: `{modeltype}`")

    assert trajs_pred.shape == (B, S, N, 2)

    ate = torch.norm(trajs_pred - trajs_gt, dim=-1)
    assert ate.shape == (B, S, N)
    ate_all = pips_utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = pips_utils.basic.reduce_masked_mean(ate, valids * good_visibility_mask)
    ate_occ = pips_utils.basic.reduce_masked_mean(ate, valids * (1.0 - good_visibility_mask))

    results = {
        "ate_all": ate_all.item(),
        "ate_vis": ate_vis.item(),
        "ate_occ": ate_occ.item(),
        "B": B, "S": S, "C": C, "H": H, "W": W, "N": N, "D": D,
        "trajectory_gt": trajs_gt.detach().clone().cpu(),
        "trajectory_pred": trajs_pred.detach().clone().cpu(),
        "valids": valids.detach().clone().cpu(),
        "visibility_gt": vis_gt.detach().clone().cpu(),
    }
    assert valids.all().item(), "FlyingThings++ always has all points valid"

    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_gt, pips_utils.improc.preprocess_color(rgbs),
                                cmap='winter', linewidth=2)

        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_pred[0:1], pips_utils.improc.preprocess_color(rgbs[0:1]),
                                cmap='spring', linewidth=2)
        gt_rgb = pips_utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_gt[0:1],
                                   torch.mean(pips_utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter',
                                   frame_id=results['ate_all'], only_return=True, linewidth=2))
        gt_black = pips_utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_gt[0:1],
                                   torch.ones_like(rgbs[0:1, 0]) * -0.5, cmap='winter', frame_id=results['ate_all'],
                                   only_return=True, linewidth=2))

        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_pred[0:1], gt_rgb[0:1], cmap='spring',
                               linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_pred[0:1], gt_black[0:1], cmap='spring',
                               linewidth=2)

        # animate_traj2ds_on_rgbs
        if modeltype == "pips":
            rgb_vis = []
            black_vis = []
            for trajs_e_ in preds_anim:
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], gt_rgb, only_return=True, cmap='coolwarm'))
                black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], gt_black, only_return=True, cmap='coolwarm'))
            sw.summ_rgbs('outputs/animated_trajs_on_black', black_vis)
            sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return results


def main(
        exp_name='flt',
        B=1,
        S=8,
        N=16,
        modeltype='pips',
        init_dir='reference_model',
        stride=4,
        log_dir='logs_test_on_flt',
        dataset_location='data/flyingthings',
        max_iters=0,  # auto-select based on dataset
        log_freq=100,
        shuffle=False,
        subset='all',
        crop_size=(384, 512),  # the raw data is 540,960
        use_augs=False,
        seed=72,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    assert (modeltype == 'pips' or modeltype == 'raft' or modeltype == 'dino')

    model_name = f"{B:d}_{S:d}_{N:d}_{modeltype:s}"
    if use_augs:
        model_name += "_A"
    model_name += f"_{exp_name:s}"
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(os.path.join(log_dir, model_name, "t"), max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    test_dataset = FlyingThingsDataset(
        dataset_location=dataset_location,
        dset='TEST', subset=subset,
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=24,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    test_iterloader = iter(test_dataloader)

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

    n_pool = 10000
    ate_all_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')

    if max_iters == 0:
        max_iters = len(test_dataloader)
    print(f'max_iters={max_iters}')

    results_list = []
    global_step = 0
    while global_step < max_iters:
        read_start_time = time.time()
        global_step += 1
        sw_t = pips_utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq / 2),
            just_gif=True
        )

        gotit = (False, False)
        while not all(gotit):
            try:
                sample, gotit = next(test_iterloader)
            except StopIteration:
                test_iterloader = iter(test_dataloader)
                sample, gotit = next(test_iterloader)

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            packed_results = run_for_sample(modeltype, model, sample, sw_t)
            for b in range(packed_results["trajectory_gt"].shape[0]):
                for n in range(packed_results["trajectory_gt"].shape[2]):
                    results_list += [{
                        "iter": global_step,
                        "video_idx": b,
                        "point_idx_in_video": n,
                        "trajectory_gt": packed_results["trajectory_gt"][b, :, n, :],
                        "trajectory_pred": packed_results["trajectory_pred"][b, :, n, :],
                        "valids": packed_results["valids"][b, :, n],
                        "visibility_gt": packed_results["visibility_gt"][b, :, n],
                    }]

        if packed_results['ate_all'] > 0:
            ate_all_pool_t.update([packed_results['ate_all']])
        if packed_results['ate_vis'] > 0:
            ate_vis_pool_t.update([packed_results['ate_vis']])
        if packed_results['ate_occ'] > 0:
            ate_occ_pool_t.update([packed_results['ate_occ']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())

        iter_time = time.time() - iter_start_time
        print(
            f'{model_name}'
            f' step={global_step:06d}/{max_iters:d}'
            f' readtime={read_time:>2.2f}'
            f' itertime={iter_time:>2.2f}'
            f' ate_all={ate_all_pool_t.mean():>2.2f}'
            f' ate_vis={ate_vis_pool_t.mean():>2.2f}'
            f' ate_occ={ate_occ_pool_t.mean():>2.2f}'
        )

    writer_t.close()
    results_list_pkl_path = os.path.join(log_dir, model_name, "results_list.pkl")
    with open(results_list_pkl_path, "wb") as f:
        print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
        pickle.dump(results_list, f)

    results_df_path = os.path.join(log_dir, model_name, "results_df.csv")
    results_df = compute_summary_df(results_list)
    results_df.to_csv(results_df_path)
    print(f"\nResults summary dataframe saved to:\n{results_df_path}\n")


if __name__ == '__main__':
    Fire(main)
