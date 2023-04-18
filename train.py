import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from fire import Fire
from tensorboardX import SummaryWriter

import pips_utils.improc
from datasets.factory import DataloaderFactory
from nets.pips import Pips
from pips_utils import saverloader
from pips_utils.basic import print_
from pips_utils.util import seed_all, get_str_formatted_time

random.seed(125)
np.random.seed(125)


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps + 100,
                                                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def run_model(model, batch, dataset_type, device, I=6, horz_flip=False, vert_flip=False, sw=None, is_train=True):
    if dataset_type == "flyingthings++":
        # flow = d['flow'].cuda().permute(0, 3, 1, 2)
        rgbs = batch['rgbs'].to(device).float()  # B, S, C, H, W
        occs = batch['occs'].to(device).float()  # B, S, 1, H, W
        masks = batch['masks'].to(device).float()  # B, S, 1, H, W
        trajectories_gt = batch['trajs'].to(device).float()  # B, S, N, 2
        visibilities_gt = batch['visibles'].to(device).float()  # B, S, N
        valids = batch['valids'].to(device).float()  # B, S, N

        batch_size, n_frames, channels, height, width = rgbs.shape
        assert (channels == 3)
        n_points = trajectories_gt.shape[2]
        assert (torch.sum(valids) == batch_size * n_frames * n_points)

        if horz_flip:  # increase the batchsize by horizontal flipping
            rgbs_flip = torch.flip(rgbs, [4])
            occs_flip = torch.flip(occs, [4])
            masks_flip = torch.flip(masks, [4])
            trajs_g_flip = trajectories_gt.clone()
            trajs_g_flip[:, :, :, 0] = width - 1 - trajs_g_flip[:, :, :, 0]
            vis_g_flip = visibilities_gt.clone()
            valids_flip = valids.clone()
            trajectories_gt = torch.cat([trajectories_gt, trajs_g_flip], dim=0)
            visibilities_gt = torch.cat([visibilities_gt, vis_g_flip], dim=0)
            valids = torch.cat([valids, valids_flip], dim=0)
            rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
            occs = torch.cat([occs, occs_flip], dim=0)
            masks = torch.cat([masks, masks_flip], dim=0)
            batch_size = batch_size * 2

        if vert_flip:  # increase the batchsize by vertical flipping
            rgbs_flip = torch.flip(rgbs, [3])
            occs_flip = torch.flip(occs, [3])
            masks_flip = torch.flip(masks, [3])
            trajs_g_flip = trajectories_gt.clone()
            trajs_g_flip[:, :, :, 1] = height - 1 - trajs_g_flip[:, :, :, 1]
            vis_g_flip = visibilities_gt.clone()
            valids_flip = valids.clone()
            trajectories_gt = torch.cat([trajectories_gt, trajs_g_flip], dim=0)
            visibilities_gt = torch.cat([visibilities_gt, vis_g_flip], dim=0)
            valids = torch.cat([valids, valids_flip], dim=0)
            rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
            occs = torch.cat([occs, occs_flip], dim=0)
            masks = torch.cat([masks, masks_flip], dim=0)
            batch_size = batch_size * 2
    else:
        rgbs, query_points, trajectories_gt, visibilities_gt = DataloaderFactory.unpack_batch(
            batch=batch,
            dataset=dataset_type,
            modeltype="pips",
            device=device,
        )
        visibilities_gt = visibilities_gt.float()
        valids = torch.ones_like(visibilities_gt)

    batch_size, n_frames, channels, height, width = rgbs.shape
    n_points = trajectories_gt.shape[2]

    preds, preds_anim, vis_e, stats = model(trajectories_gt[:, 0], rgbs, coords_init=None, iters=I,
                                            trajs_g=trajectories_gt, vis_g=visibilities_gt, valids=valids, sw=sw,
                                            is_train=is_train)
    seq_loss, vis_loss, ce_loss = stats

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    total_loss += seq_loss.mean()
    total_loss += vis_loss.mean() * 10.0
    total_loss += ce_loss.mean()

    ate = torch.norm(preds[-1] - trajectories_gt, dim=-1)  # B, S, N
    ate_all = pips_utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = pips_utils.basic.reduce_masked_mean(ate, valids * visibilities_gt)
    ate_occ = pips_utils.basic.reduce_masked_mean(ate, valids * (1.0 - visibilities_gt))

    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
        'seq': seq_loss.mean().item(),
        'vis': vis_loss.mean().item(),
        'ce': ce_loss.mean().item()
    }

    if sw is not None and sw.save_this:
        trajs_e = preds[-1]

        pad = 50
        rgbs = F.pad(rgbs.reshape(batch_size * n_frames, 3, height, width), (pad, pad, pad, pad), 'constant', 0).reshape(batch_size, n_frames, 3, height + pad * 2,
                                                                                                width + pad * 2)
        # occs = F.pad(occs.reshape(B * S, 1, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 1, H + pad * 2,
        #                                                                                         W + pad * 2)
        # masks = F.pad(masks.reshape(B * S, 1, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 1, H + pad * 2,
        #                                                                                           W + pad * 2)
        trajs_e = trajs_e + pad
        trajectories_gt = trajectories_gt + pad

        # occs_ = occs[0].reshape(S, -1)
        # counts_ = torch.max(occs_, dim=1)[0]
        # print('counts_', counts_)
        # sw.summ_rgbs('0_inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # sw.summ_oneds('0_inputs/occs', occs.unbind(1), frame_ids=counts_)
        # sw.summ_oneds('0_inputs/masks', masks.unbind(1), frame_ids=counts_)
        # sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs2', trajs_g[0:1], visibilities_gt[0:1], utils.improc.preprocess_color(rgbs[0:1]), valids=valids[0:1], cmap='winter')
        sw.summ_traj2ds_on_rgbs2('0_inputs/trajs_on_rgbs2', trajectories_gt[0:1], visibilities_gt[0:1],
                                 pips_utils.improc.preprocess_color(rgbs[0:1]))

        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb', trajectories_gt[0:1],
                               torch.mean(pips_utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter')

        for b in range(batch_size):
            sw.summ_traj2ds_on_rgb('0_batch_inputs/trajs_g_on_rgb_%d' % b, trajectories_gt[b:b + 1],
                                   torch.mean(pips_utils.improc.preprocess_color(rgbs[b:b + 1]), dim=1), cmap='winter')

        # sw.summ_traj2ds_on_rgbs2('2_outputs/trajs_e_on_rgbs', trajs_e[0:1], torch.sigmoid(vis_e[0:1]), utils.improc.preprocess_color(rgbs[0:1]), cmap='spring')
        # sw.summ_traj2ds_on_rgbs('2_outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring')

        gt_rgb = pips_utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajectories_gt[0:1], torch.mean(
            pips_utils.improc.preprocess_color(rgbs[0:1]), dim=1), valids=valids[0:1], cmap='winter',
                                                                           frame_id=metrics['ate_all'],
                                                                           only_return=True))
        # gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, valids=valids[0:1], cmap='winter', frame_id=metrics['ate_all'], only_return=True))
        sw.summ_traj2ds_on_rgb('2_outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring')
        # sw.summ_traj2ds_on_rgb('2_outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring')

        if True:  # this works but it's a bit expensive
            rgb_vis = []
            # black_vis = []
            for trajs_e in preds_anim:
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1] + pad, gt_rgb, only_return=True, cmap='spring'))
                # black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1]+pad, gt_black, only_return=True, cmap='spring'))
            sw.summ_rgbs('2_outputs/animated_trajs_on_rgb', rgb_vis)
            # sw.summ_rgbs('2_outputs/animated_trajs_on_black', black_vis)

    return total_loss, metrics


def main(
        exp_name='debug',
        # training
        B=4,  # batchsize
        S=8,  # seqlen of the data/model
        N=768,  # number of particles to sample from the data
        horz_flip=True,  # this causes B*=2
        vert_flip=True,  # this causes B*=2
        stride=8,  # spatial stride of the model
        I=4,  # inference iters of the model
        crop_size=(384, 512),  # the raw data is 540,960
        # crop_size=(256,384), # the raw data is 540,960
        use_augs=True,  # resizing/jittering/color/blur augs
        # dataset
        dataset_type='flyingthings++',
        dataset_location='data/flyingthings',
        subset_train='all-train',
        subset_valid='all-test',
        shuffle=True,  # dataset shuffling
        dataloader_workers=None,
        # optimization
        lr=5e-4,
        grad_acc=1,
        max_iters=200000,
        use_scheduler=True,
        # summaries
        log_dir='logs/my_pips/logs_train',
        log_freq=4000,
        val_freq=2000,
        # saving/loading
        ckpt_dir='logs/my_pips/checkpoints',
        save_freq=1000,
        keep_latest=1,
        init_dir='',
        load_optimizer=False,
        load_step=False,
        ignore_load=None,
        # cuda
        device_ids=[0],
        seed=72,
):
    # the idea in this file is to train a PIPs model (nets/pips.py) in flyingthings++

    seed_all(seed)

    ## autogen a descriptive name
    experiment_name = f"{B:d}"
    experiment_name += f"_{S:d}"
    experiment_name += f"_{N}"
    experiment_name += f"_{seed}"
    experiment_name += f"_{dataset_type}"
    experiment_name += f"_train={subset_train}"
    experiment_name += f"_val={subset_valid}"

    if horz_flip and vert_flip:
        experiment_name += "%dhv" % (B * 4)
    elif horz_flip:
        experiment_name += "%dh" % (B * 2)
    elif vert_flip:
        experiment_name += "%dv" % (B * 2)
    else:
        experiment_name += "%d" % (B)
    if grad_acc > 1:
        experiment_name += "x%d" % grad_acc
    experiment_name += "_I%d" % (I)

    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    experiment_name += "_%s" % lrn
    if use_augs:
        experiment_name += "_A"
    experiment_name += "_%s" % exp_name
    experiment_name += f"_{get_str_formatted_time()}"
    print(f"experiment_name={experiment_name}")

    assert (crop_size[0] % 128 == 0)
    assert (crop_size[1] % 128 == 0)

    device = 'cuda:%d' % device_ids[0]

    ckpt_dir = '%s/%s' % (ckpt_dir, experiment_name)
    writer_t = SummaryWriter(log_dir + '/' + experiment_name + '/t', max_queue=10, flush_secs=60)
    if val_freq > 0:
        writer_v = SummaryWriter(log_dir + '/' + experiment_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    if dataloader_workers is None:
        dataloader_workers = 16 * len(device_ids)

    train_dataloader = DataloaderFactory.get_dataloader(
        dataset_type, dataset_location, subset_train,
        query_mode=None, pips_window=S, flt_use_augs=use_augs,
        flt_crop_size=crop_size, n_points=N, batch_size=B,
        shuffle=shuffle, dataloader_workers=dataloader_workers)
    train_iterloader = iter(train_dataloader)

    if val_freq > 0:
        print('not using augs in val')
        val_dataloader = DataloaderFactory.get_dataloader(
            dataset_type, dataset_location, subset_valid,
            query_mode=None, pips_window=S, flt_use_augs=False,
            flt_crop_size=crop_size, n_points=N, batch_size=B,
            shuffle=shuffle, dataloader_workers=dataloader_workers)
        val_iterloader = iter(val_dataloader)

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters // grad_acc, model.parameters())
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-7)

    global_step = 0
    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, device, optimizer, ignore_load=ignore_load)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, device, ignore_load=ignore_load)
        else:
            _ = saverloader.load(init_dir, model.module, device, ignore_load=ignore_load)
            global_step = 0
    requires_grad(parameters, True)
    model.train()

    n_pool = 100
    loss_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ce_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    vis_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    seq_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ate_all_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = pips_utils.misc.SimplePool(n_pool, version='np')
    if val_freq > 0:
        loss_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        ce_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        vis_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        seq_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        ate_all_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        ate_vis_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')
        ate_occ_pool_v = pips_utils.misc.SimplePool(n_pool, version='np')

    while global_step < max_iters:
        global_step += 1

        iter_read_time = 0.0
        iter_start_time = time.time()
        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step == grad_acc - 1:
                sw_t = pips_utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=5,
                    scalar_freq=int(log_freq / 2),
                    just_gif=True)
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

            # TODO read_time not computed correctly for grad_acc
            iter_read_time += time.time() - read_start_time

            total_loss, metrics = run_model(model, sample, dataset_type, device, I, horz_flip, vert_flip, sw_t,
                                            is_train=True)
            total_loss.backward()

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])
        if metrics['ce'] > 0:
            ce_pool_t.update([metrics['ce']])
        if metrics['vis'] > 0:
            vis_pool_t.update([metrics['vis']])
        if metrics['seq'] > 0:
            seq_pool_t.update([metrics['seq']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())
        sw_t.summ_scalar('pooled/ce', ce_pool_t.mean())
        sw_t.summ_scalar('pooled/vis', vis_pool_t.mean())
        sw_t.summ_scalar('pooled/seq', seq_pool_t.mean())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        if val_freq > 0 and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = pips_utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq / 2),
                just_gif=True)

            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(train_dataloader)
                sample = next(val_iterloader)

            with torch.no_grad():
                total_loss, metrics = run_model(model, sample, dataset_type, device, I, horz_flip, vert_flip, sw_v, is_train=False)

            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

            if metrics['ate_all'] > 0:
                ate_all_pool_v.update([metrics['ate_all']])
            if metrics['ate_vis'] > 0:
                ate_vis_pool_v.update([metrics['ate_vis']])
            if metrics['ate_occ'] > 0:
                ate_occ_pool_v.update([metrics['ate_occ']])
            if metrics['ce'] > 0:
                ce_pool_v.update([metrics['ce']])
            if metrics['vis'] > 0:
                vis_pool_v.update([metrics['vis']])
            if metrics['seq'] > 0:
                seq_pool_v.update([metrics['seq']])
            sw_v.summ_scalar('pooled/ate_all', ate_all_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_vis', ate_vis_pool_v.mean())
            sw_v.summ_scalar('pooled/ate_occ', ate_occ_pool_v.mean())
            sw_v.summ_scalar('pooled/ce', ce_pool_v.mean())
            sw_v.summ_scalar('pooled/vis', vis_pool_v.mean())
            sw_v.summ_scalar('pooled/seq', seq_pool_v.mean())
            model.train()

        if np.mod(global_step, save_freq) == 0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, keep_latest=keep_latest)

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            experiment_name, global_step, max_iters, iter_read_time, iter_time, total_loss.item()))

    writer_t.close()
    if val_freq > 0:
        writer_v.close()


if __name__ == '__main__':
    Fire(main)
