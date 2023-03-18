"""
Evaluate on Head Tracking in CroHD. Supports evaluating PIPs, RAFT, and DINO.

PIPs vis: 4.57, PIPs occ: 7.71 TODO: numbers not verified

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
from crohddataset import CrohdDataset, prep_sample
from nets.pips import Pips
from nets.raftnet import Raftnet
from pips_utils.figures import compute_summary_df, figure1, figure2
from pips_utils.util import ensure_dir
from test_on_flt import run_for_sample

device = 'cuda'
random.seed(125)
np.random.seed(125)


def main(
        exp_name='crohd',
        B=1,
        S=8,
        N=16,
        modeltype='pips',
        init_dir='reference_model',
        req_occlusion=True,
        stride=4,
        log_dir='logs_test_on_crohd',
        dataset_location='data/head_tracking',
        max_iters=0,  # auto-select based on dataset
        log_freq=100,
        shuffle=False,
        subset='all',
        use_augs=False,
        S_stride=3,  # subsample the frames this much
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
    if req_occlusion:
        model_name += "_occ"
    else:
        model_name += "_vis"
    model_name += f"_{exp_name:s}"
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(os.path.join(log_dir, model_name, "t"), max_queue=10, flush_secs=60)

    dataset = CrohdDataset(seqlen=S * S_stride, dataset_root=dataset_location)
    test_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=12)
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
            just_gif=True,
        )

        returned_early = True
        while returned_early:
            try:
                sample = next(test_iterloader)
            except StopIteration:
                test_iterloader = iter(test_dataloader)
                sample = next(test_iterloader)
            sample, returned_early = prep_sample(sample, N, S_stride, req_occlusion)

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            packed_results = run_for_sample(modeltype, model, sample, sw_t, dataset="crohd")
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

    figures_dir = os.path.join(log_dir, model_name, "figures")
    ensure_dir(figures_dir)
    results_df["name"] = modeltype
    figure1(results_df, figures_dir)
    figure2(results_df, figures_dir)


if __name__ == '__main__':
    Fire(main)
