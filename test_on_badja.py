import time
import argparse
import numpy as np
import timeit
import matplotlib
#import tensorflow as tf
# import scipy.misc
import io
import os
import math
from PIL import Image
matplotlib.use('Agg') # suppress plot showing

import matplotlib.pyplot as plt

import matplotlib.animation as animation
import cv2
import saverloader
import sys 

#from raft_core.raft import RAFT
# from relative_perceiver import RelativePerceiver
# from sparse_relative_perceiver import SparseRelativePerceiver
# from nets.graph_raft import GraphRaft
#from nets.st_graph_raft import StGraphRaft
# from nets.st_spraft import StSpRaft
# from nets.st_graph_raft import StGraphRaft
# # from nets.mraft import Mraft
# from nets.praft import Praft
#from nets.mpraft import Mpraft
# from perceiver_graph import PerceiverGraph
# from relative_mlp import RelativeMlp
#from nets.dofperionet import Dofperionet

from nets.raftnet import Raftnet
from nets.singlepoint import Singlepoint

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

# import nets.relation_ebm as relation_ebm
# import nets.transformer_forecaster as transformer_forecaster
# import transformer_forecaster

# import nets.transformer_ebm as transformer_ebm
# # from nets.improved_cd_models import CelebAModel, ResNetModel
# import nets.conv1d_ebm
# import nets.traj_mlp_ebm
# import nets.encoder2d
# import nets.sparse_invar_encoder2d
# # import nets.improved_cd_models
# import nets.raftnet
# import nets.seg2dnet
# import nets.segpointnet

import utils.py
#import utils.box
import utils.misc
import utils.improc
import utils.vox
import utils.grouping
from tqdm import tqdm
import random
import glob
# import color2d

# import detectron2
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.modeling import build_model

from utils.basic import print_, print_stats


# import relation_model

import badjadataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

# from raft_core.utils import InputPadder
                

# import inputs

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

def run_static(d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    # print('rgbs', rgbs.shape)
    # print('trajs', trajs.shape)

    # S = min(8,S)
    # rgbs = rgbs[:,:S]
    # segs = segs[:,:S]
    # trajs_g = trajs_g[:,:S]
    # visibles = visibles[:,:S]
    
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    # print_stats('segs', segs)
    segs = (segs > 0).float()
    assert(B==1)
    accs = []
    # for s0 in range(S): # source frame
    for s0 in range(1): # source frame
        # for s1 in range(1,S): # target frame
        for s1 in range(S): # target frame
            if not s0==s1:
                for n in range(N):
                    vis = visibles[0,s1,n]
                    if vis > 0:
                        coord_e = trajs_g[0,s0,n] # 2
                        coord_g = trajs_g[0,s1,n] # 2
                        dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                        # print_('dist', dist)
                        area = torch.sum(segs[0,s1])
                        # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                        thr = 0.2 * torch.sqrt(area)
                        correct = (dist < thr).float()
                        # print_('correct', correct)
                        accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    # print('pck', pck.item())

    if sw is not None and sw.save_this:
        label_colors = utils.improc.get_n_colors(N)
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()

            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            # rgb = (torch.mean(rgbs[:,s] * 0.5, dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)

        prep_rgbs = utils.improc.preprocess_color(rgbs)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])
    
    return metrics

def run_singlepoint(singlepoint, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('rgbs', rgbs.shape)

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    # N = 2

    # S = min(8,S)
    # rgbs = rgbs[:,:S]
    # segs = segs[:,:S]
    # trajs_g = trajs_g[:,:S]
    # visibles = visibles[:,:S]

    # if S < 8:
    #     rgbs = torch.cat([rgbs, rgbs[:,-1].unsqueeze(1).repeat(1,8-S,1,1,1)], dim=1)
    #     segs = torch.cat([segs, segs[:,-1].unsqueeze(1).repeat(1,8-S,1,1,1)], dim=1)
    #     trajs_g = torch.cat([trajs_g, trajs_g[:,-1].unsqueeze(1).repeat(1,8-S,1,1)], dim=1)
    #     visibles = torch.cat([visibles, visibles[:,-1].unsqueeze(1).repeat(1,8-S,1)], dim=1)
    #     S = 8
    
    # print('rgbs', rgbs.shape)
    # print('trajs_g', trajs_g.shape)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    # H_, W_ = 384, 512
    # H_, W_ = 512, 768
    # H_, W_ = 512//2, 768//2
    # H_, W_ = 496, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    segs = (segs > 0).float()
    assert(B==1)

    # print_stats('rgbs_', rgbs_)

    xy0 = trajs_g[:,0] # B, N, 2
    # what i want now is:
    # let's focus on a single pixel,
    # and track it indefinitely
    assert(S >= 8)

    # input()

    trajs_e = torch.zeros_like(trajs_g)
    # n = 0
    # if True:
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros_like(trajs_g[:,:,n]) # B, S, 2
        # xy0_n = trajs_g[:,0,n] # B, 1, 2
        traj_e[:,0] = trajs_g[:,0,n] # B, 1, 2  # set first position to gt
        feat_init = None
        while not done:
            end_frame = cur_frame + 8
            # print('cur_frame', cur_frame)
            # print('end_frame', end_frame)

            rgb_seq = rgbs[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            # print('S_local', S_local)
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)
            # print('rgb_seq (%d:%d)' % (cur_frame, end_frame), rgb_seq.shape)

            outs = singlepoint(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            # print_('vis', vis.squeeze())
            xys = preds[-1].reshape(1, 8, 2)
            # import ipdb; ipdb.set_trace()
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]
            # for si in range(end_frame,curr_frame,-1):
            #     if vis[0,si] > 0.99:

            # sw.summ_traj2ds_on_rgb('kp_%d/traj_from_%d' % (n, cur_frame), xys.reshape(1, 8, 1, 2), utils.improc.preprocess_color(rgb_seq), cmap='spring')
            # sw.summ_traj2ds_on_rgbs('kp_%d/traj_from_%d' % (n, cur_frame), xys.reshape(1, 8, 1, 2), utils.improc.preprocess_color(rgb_seq), cmap='spring')
            
            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())
            # si = 7

            # input()

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    label_colors = utils.improc.get_n_colors(N)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    
    if sw is not None and sw.save_this:
        if True:
            for n in range(N):
                if visibles[0,0,n] > 0:
                    print('visualizing kp %d' % n)
                    # sw.summ_traj2ds_on_rgbs('kp_outputs_%02d/trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
                    sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
                    
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')
        # sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb3', trajs_e[0:1,0:1], prep_rgbs[0:1,0], cmap='plasma')
        # sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb4', trajs_e[0:1], prep_rgbs[0:1,0], cmap='plasma')

        if False:
            kp_vis = []
            for s in range(S):
                kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
                kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
                kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
                kp[kp_any==0] = rgb[kp_any==0]
                kp_vis.append(kp)
            sw.summ_rgbs('outputs/kp_vis', kp_vis)

    assert(B==1)
    accs = []
    for s1 in range(S): # target frame
        for n in range(N):
            vis = visibles[0,s1,n]
            if vis > 0:
                coord_e = trajs_e[0,s1,n] # 2
                coord_g = trajs_g[0,s1,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,s1])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                # print_('correct', correct)
                accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    # print('pck', pck.item())
    
    # metrics['pck'] = 0
    return metrics
        
    # trajs_e = xys.reshape(1, -1, S, 2).permute(0, 2, 1, 3) # 1, S, 1, 2

    # rgbs_ = rgbs[:,:S]
    # xy0 = trajs_g[:,0]
    # outs = singlepoint(xy0.reshape(-1, 1, 2), rgbs_, iters=12)
    # preds = outs[0]
    # xys = preds[-1]
    # trajs_e = xys.reshape(1, -1, S, 2).permute(0, 2, 1, 3) # 1, S, N, 2
    # # print('trajs_e', trajs_e.shape)

    acc = []
    for s0 in range(1): # source frame
        # for s1 in range(1,S): # target frame
        for s1 in range(S): # target frame
            coord_e = trajs_e[:,s1] # B, N, 2
            if not s0==s1:
                coord_g = trajs_g[:,s1] # B, N, 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=2))
                # print_('dist', dist)
                area = torch.sum(segs[:,s1])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                # print_('correct', correct)
                acc.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(acc)) * 100.0
    metrics['pck'] = pck.item()
    # print('pck', pck.item())


    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')

    return metrics

def run_raft(raft, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    
    # S = min(8,S)
    # rgbs = rgbs[:,:S]
    # segs = segs[:,:S]
    # trajs_g = trajs_g[:,:S]
    # visibles = visibles[:,:S]

    # if S < 8:
    #     rgbs = torch.cat([rgbs, rgbs[:,-1].unsqueeze(1).repeat(1,8-S,1,1,1)], dim=1)
    #     segs = torch.cat([segs, segs[:,-1].unsqueeze(1).repeat(1,8-S,1,1,1)], dim=1)
    #     trajs_g = torch.cat([trajs_g, trajs_g[:,-1].unsqueeze(1).repeat(1,8-S,1,1)], dim=1)
    #     visibles = torch.cat([visibles, visibles[:,-1].unsqueeze(1).repeat(1,8-S,1)], dim=1)
    #     S = 8
    
    # print('rgbs', rgbs.shape)
    # print('trajs_g', trajs_g.shape)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    # H_, W_ = 512, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    # print_stats('segs', segs)
    segs = (segs > 0).float()
    assert(B==1)
    
    # rgbs_ = rgbs[:,:S]
    # xy0 = trajs_g[:,0]
    # outs = singlepoint(xy0.reshape(-1, 1, 2), rgbs_, iters=12)
    # preds = outs[0]
    # xys = preds[-1]
    # trajs_e = xys.reshape(1, -1, S, 2).permute(0, 2, 1, 3) # 1, S, N, 2
    # print('trajs_e', trajs_e.shape)


    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

    flows_e = []
    # for s in range(S-1):
    # print('S', S)
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        # print('rgb0', rgb0.shape)
        rgb1 = prep_rgbs[:,s+1]

        flow, _ = raft(rgb0, rgb1, iters=32)

        # # print('clamping')
        # flow = flow.clamp(min=-12, max=12)

        # rgb0_, rgb1_ = padder.pad(rgb0, rgb1)
        # # print('rgb0_', rgb0_.shape)
        
        # flow_, _ = raft(rgb0_, rgb1_, iters=12)
        # print('flow_', flow_.shape)
        # flow = padder.unpad(flow_)
        # print('flow', flow.shape)
        
        flows_e.append(flow)
        # let's backwarp to see if things work out
        
        # if sw is not None and sw.save_this:
        #     rgb0_e = utils.samp.backwarp_using_2d_flow(rgb1, flow)
        #     sw.summ_flow('flows_e/flow_%d' % s, flow, clip=100)
        #     sw.summ_rgbs('backwarps_e/rgbs_aligned_%d' % s, [rgb0, rgb0_e])
        
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    coord0 = trajs_g[:,0] # B, N, 2
    # print('coord0', coord0.shape)
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2
    # print('trajs_e', trajs_e.shape)


    assert(B==1)
    accs = []
    # for s0 in range(S): # source frame
    for s0 in range(1): # source frame
        # for s1 in range(1,S): # target frame
        for s1 in range(S): # target frame
            if not s0==s1:
                for n in range(N):
                    vis = visibles[0,s1,n]
                    if vis > 0:
                        coord_e = trajs_e[0,s1,n] # 2
                        coord_g = trajs_g[0,s1,n] # 2
                        dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                        # print_('dist', dist)
                        area = torch.sum(segs[0,s1])
                        # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                        thr = 0.2 * torch.sqrt(area)
                        correct = (dist < thr).float()
                        # print_('correct', correct)
                        accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    
    # acc = []
    # for s0 in range(1): # source frame
    #     # for s1 in range(1,S): # target frame
    #     for s1 in range(S): # target frame
    #         coord_e = trajs_e[:,s1] # B, N, 2
    #         if not s0==s1:
    #             coord_g = trajs_g[:,s1] # B, N, 2
    #             dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=2))
    #             # print_('dist', dist)
    #             area = torch.sum(segs[:,s1])
    #             # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
    #             thr = 0.2 * torch.sqrt(area)
    #             correct = (dist < thr).float()
    #             # print_('correct', correct)
    #             acc.append(correct)
    # # assert(len(acc) == S*(S-1))
    # pck = torch.mean(torch.stack(acc)) * 100.0
    # metrics['pck'] = pck.item()
    # # print('pck', pck.item())

    label_colors = utils.improc.get_n_colors(N)

    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))

        for n in range(N):
            if visibles[0,0,n] > 0:
                sw.summ_traj2ds_on_rgbs('outputs/kp%d_trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
        
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()

            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            # rgb = (torch.mean(rgbs[:,s] * 0.5, dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')

    return metrics


def run_raft_skip(raft, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    # print('file0', file0)
    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal
    # print('animal', animal)
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    # print_stats('segs', segs)
    segs = (segs > 0).float()
    assert(B==1)


    rgbs_skip = []
    segs_skip = []
    trajs_g_skip = []
    visibles_skip = []
    for s in range(S):
        # if torch.sum(segs[:,s]) > 0:
        if torch.sum(visibles[:,s]) > 0:
            rgbs_skip.append(rgbs[:,s])
            segs_skip.append(segs[:,s])
            trajs_g_skip.append(trajs_g[:,s])
            visibles_skip.append(visibles[:,s])
    rgbs = torch.stack(rgbs_skip, dim=1)
    segs = torch.stack(segs_skip, dim=1)
    trajs_g = torch.stack(trajs_g_skip, dim=1)
    visibles = torch.stack(visibles_skip, dim=1)
    S = rgbs.shape[1]


    prep_rgbs = utils.improc.preprocess_color(rgbs)

    flows_e = []
    # for s in range(S-1):
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        # print('rgb0', rgb0.shape)
        rgb1 = prep_rgbs[:,s+1]

        flow, _ = raft(rgb0, rgb1, iters=32)

        # rgb0_, rgb1_ = padder.pad(rgb0, rgb1)
        # # print('rgb0_', rgb0_.shape)
        
        # flow_, _ = raft(rgb0_, rgb1_, iters=12)
        # print('flow_', flow_.shape)
        # flow = padder.unpad(flow_)
        # print('flow', flow.shape)
        
        flows_e.append(flow)
        # let's backwarp to see if things work out
        
        # if sw is not None and sw.save_this:
        #     rgb0_e = utils.samp.backwarp_using_2d_flow(rgb1, flow)
        #     sw.summ_flow('flows_e/flow_%d' % s, flow, clip=100)
        #     sw.summ_rgbs('backwarps_e/rgbs_aligned_%d' % s, [rgb0, rgb0_e])
        
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    coord0 = trajs_g[:,0] # B, N, 2
    # print('coord0', coord0.shape)
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2
    # print('trajs_e', trajs_e.shape)


    assert(B==1)
    accs = []
    # for s0 in range(S): # source frame
    for s0 in range(1): # source frame
        # for s1 in range(1,S): # target frame
        for s1 in range(S): # target frame
            if not s0==s1:
                for n in range(N):
                    vis = visibles[0,s1,n]
                    if vis > 0:
                        coord_e = trajs_e[0,s1,n] # 2
                        coord_g = trajs_g[0,s1,n] # 2
                        dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                        # print_('dist', dist)
                        area = torch.sum(segs[0,s1])
                        # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                        thr = 0.2 * torch.sqrt(area)
                        correct = (dist < thr).float()
                        # print_('correct', correct)
                        accs.append(correct)
    # assert(len(acc) == S*(S-1))
    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()
    
    # acc = []
    # for s0 in range(1): # source frame
    #     # for s1 in range(1,S): # target frame
    #     for s1 in range(S): # target frame
    #         coord_e = trajs_e[:,s1] # B, N, 2
    #         if not s0==s1:
    #             coord_g = trajs_g[:,s1] # B, N, 2
    #             dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=2))
    #             # print_('dist', dist)
    #             area = torch.sum(segs[:,s1])
    #             # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
    #             thr = 0.2 * torch.sqrt(area)
    #             correct = (dist < thr).float()
    #             # print_('correct', correct)
    #             acc.append(correct)
    # # assert(len(acc) == S*(S-1))
    # pck = torch.mean(torch.stack(acc)) * 100.0
    # metrics['pck'] = pck.item()
    # # print('pck', pck.item())

    label_colors = utils.improc.get_n_colors(N)

    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()

            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            # rgb = (torch.mean(rgbs[:,s] * 0.5, dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')

    return metrics



def run_model(model, d, sw):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {
        'epe': 0,
        'epe_occ': 0,
        'epe_inv': 0,
        'epe_inv2inv': 0,
    }

    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N
    
    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    print('rgbs', rgbs.shape)
    # print('trajs', trajs.shape)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_

    label_colors = utils.improc.get_n_colors(N)
    
    kp_vis = []
    for s in range(S):
        kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
        kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
        kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()

        kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
        # rgb = (torch.mean(rgbs[:,s] * 0.5, dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
        rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
        # print('rgb', rgb.shape)
        kp[kp_any==0] = rgb[kp_any==0]
        kp_vis.append(kp)
    sw.summ_rgbs('inputs/kp_vis', kp_vis)
    
    prep_rgbs = utils.improc.preprocess_color(rgbs)
    sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
    sw.summ_oneds('inputs/segs', segs.unbind(1))
    sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
    sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])
    
    return total_loss, metrics
    
    flows_g = d['flows'].cuda().float() # B, S, 2, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    paths = d['paths']
    print('paths', paths)


    if sw is not None and sw.save_this:
        # let's see these

        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        # sw.summ_oneds('inputs/occs', occs.unbind(1), norm=False)
        sw.summ_oneds('inputs/occs', occs.unbind(1))
        # sw.summ_flows('inputs/occs', occs.unbind(1), norm=False)

        for s in range(S-1):
            flow01 = flows_g[:,s]
            rgb0 = prep_rgbs[:,s]
            rgb1 = prep_rgbs[:,s+1]
            sw.summ_flow('flows_g/flow_%d' % s, flow01, clip=100)
            # let's backwarp to see if things work out
            rgb0_e = utils.samp.backwarp_using_2d_flow(rgb1, flow01)
            sw.summ_rgbs('backwarps_g/rgbs_aligned_%d' % s, [rgb0, rgb0_e])

        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter')
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1,:2], prep_rgbs[0:1,0], cmap='winter')

    # return total_loss, metrics
            

    # i want to run mpraft once for every pixel, and measure epe using the flow estimate
    if False:

        rgbs_ = rgbs.reshape(B*S, C, H, W)
        H_, W_ = 320, 512
        sy = H_/H
        sx = W_/W
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        
        stride = 4
        H2, W2 = int(H_/stride), int(W_/stride)
        xy = utils.basic.gridcloud2d(1, H2, W2).reshape(H2*W2, 2)*stride


        xy_list = torch.split(xy, 128, dim=0)
        trajs_e = []
        print('len(xy_list)', len(xy_list))
        for xy0 in xy_list:
            # print('xy0', xy0.shape)
            outs = model(xy0.reshape(-1, 1, 2), rgbs_.reshape(1, S, 3, H_, W_), iters=12)
            preds = outs[0]
            xys = preds[-1]
            xys = xys.reshape(1, -1, S, 2).permute(0, 2, 1, 3) # 1, S, N, 2
            # xys[:,:,:,0] /= sx
            # xys[:,:,:,1] /= sy
            trajs_e.append(xys)
            sys.stdout.write('.')
            sys.stdout.flush()
        trajs_e = torch.cat(trajs_e, dim=2) # 1, S, N, 2
        print('trajs_e', trajs_e.shape)

        flow_e = trajs_e[:,1] - trajs_e[:,0]
        flow_e = flow_e.reshape(1, H2, W2, 2).permute(0, 3, 1, 2)
        sw.summ_flow('outputs/flow_e', flow_e)
    else:
        


        pad_ht = (((H // 8) + 1) * 8 - H) % 8
        pad_wd = (((W // 8) + 1) * 8 - W) % 8
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        rgbs_ = list(rgbs.unbind(1))
        rgbs_ = [F.pad(x, _pad, mode='replicate') for x in rgbs_]
        rgbs_ = torch.stack(rgbs_, dim=1)
        # print('rgbs_[0]', rgbs_[0].shape)
        # rgb0_, rgb1_ = padder.pad(rgb0, rgb1)

        _, _, _, H_, W_ = rgbs_.shape

        # xy_ = utils.basic.gridcloud2d(1, H_, W_).reshape(H_*W_, 2)
        # print('xy_', xy_.shape)

        # flow_g = flows_g[:,0] # B, 2, H, W

        # occ_map = occs[:,0]
        # print_stats('occ_map', occ_map)
        # occ_map_ = F.pad(occ_map, _pad, mode='constant', value=0.0)
        # flow_g_ = F.pad(flow_g, _pad, mode='constant', value=0.0)
        # print('rgbs_', rgbs_.shape)
        # print('occ_map_', occ_map_.shape)

        # xy = xy_.reshape(-1, 2)[occ_map_.reshape(-1) > 0] # ?, 2
        # gt = flow_g_.permute(0, 2, 3, 1).reshape(-1, 2)[occ_map_.reshape(-1) > 0] # ?, 2


        # print_stats('trajs_g', trajs_g)
        # print('trajs_g', trajs_g.shape)
        trajs_g[:,:,:,0] += pad_wd//2
        trajs_g[:,:,:,1] += pad_ht//2

        xy = trajs_g[0,0] # N, 2
        # xy[:,0] += pad_wd
        # xy[:,1] += pad_ht
        xy_list = torch.split(xy, 128, dim=0)
        trajs_e = []
        print('len(xy_list)', len(xy_list))
        for xy0 in xy_list:
            # print('xy0', xy0.shape)
            outs = model(xy0.reshape(-1, 1, 2), rgbs_.reshape(1, S, 3, H_, W_), iters=12)
            preds = outs[0]
            xys = preds[-1]
            xys = xys.reshape(1, -1, S, 2).permute(0, 2, 1, 3) # 1, S, N, 2
            # xys[:,:,:,0] /= sx
            # xys[:,:,:,1] /= sy
            trajs_e.append(xys)
            sys.stdout.write('.')
            sys.stdout.flush()
        trajs_e = torch.cat(trajs_e, dim=2) # 1, S, N, 2
        print('trajs_e', trajs_e.shape)

        epe = torch.mean(torch.sqrt(torch.sum((trajs_e[:,-1] - trajs_g[:,-1])**2, dim=-1, keepdim=True)))
        metrics['epe'] = epe
        # metrics['epe_occ'] = epe_occ

        # rgb0 = prep_rgbs[:,0]
        # rgb1 = prep_rgbs[:,1]
        # rgb0_e = utils.samp.backwarp_using_2d_flow(rgb1, flow_e)
        if sw is not None and sw.save_this:
            sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs_[0:1]), cmap='spring', frame_ids=[epe.item()]*S)
            sw.summ_traj2ds_on_rgbs('outputs/trajs_g_on_rgbs', trajs_g[0:1], utils.improc.preprocess_color(rgbs_[0:1]), cmap='winter', frame_ids=[epe.item()]*S)

            sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], utils.improc.preprocess_color(rgbs_[0:1,0]), cmap='spring', frame_id=epe.item())
            sw.summ_traj2ds_on_rgb('outputs/trajs_g_on_rgb', trajs_g[0:1], utils.improc.preprocess_color(rgbs_[0:1,0]), cmap='winter', frame_id=epe.item())
            # sw.summ_traj2ds_on_rgbs('outputs/gt_flow_on_rgb', (flow_g_ + flow_0.permute(1,0).reshape(1, 2, H_, W_)), utils.improc.preprocess_color(rgbs_[0:1,:2]), cmap='spring')

    return total_loss, metrics


def prep_frame_for_dino(img, scale_size=[192]):
    """
    read a single frame & preprocess
    """
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()

    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x
    
    img = color_normalize(img)
    return img, ori_h, ori_w

def get_feats_from_dino(model, frame):
    # batch version of the other func
    B = frame.shape[0]
    h, w = int(frame.shape[2] / model.patch_embed.patch_size), int(frame.shape[3] / model.patch_embed.patch_size)
    out = model.get_intermediate_layers(frame.cuda(), n=1)[0] # B, 1+h*w, dim
    dim = out.shape[-1]
    out = out[:, 1:, :]  # discard the [CLS] token
    outmap = out.permute(0, 2, 1).reshape(B, dim, h, w)
    return out, outmap, h, w

def restrict_neighborhood(h, w):
    size_mask_neighborhood = 12
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)

def label_propagation(h, w, feat_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    # print('feat_tar', feat_tar.shape)
    # print('feat_sources', feat_sources.shape)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)

    size_mask_neighborhood = 0
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h*w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    topk = 5
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    
    return seg_tar, mask_neighborhood

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def run_dino(dino, d, sw):
    import copy
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    patch_size = 8

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    # H_, W_ = 320, 512
    # H_, W_ = 384, 512
    # H_, W_ = 496, 768
    H_, W_ = 512, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    segs = (segs > 0).float()
    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    import copy
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run dino
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1,2,0).detach().cpu().numpy(), scale_size=[H_])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0) # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0 
        featmaps = []
        while idx < S:
            end_id = min(S, idx+bs)
            _, featmaps_cur, h, w = get_feats_from_dino(dino, prep_rgbs[idx:end_id]) # S, C, h, w
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0) # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    first_seg = torch.zeros((1, N, H_//patch_size, W_//patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h*w) # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    trajs_e[0,0] = trajs_g[0,0]
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h*w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = visibles[0,cnt,n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / frame_tar_avg[n][nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
            else:
                # stay where it was
                # coord_e = trajs_g[0,0,n]
                coord_e = trajs_e[0,cnt-1,n]
            trajs_e[0, cnt, n] = coord_e
            if vis > 0:
                coord_g = trajs_g[0,cnt,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,cnt])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

        # if len(accs) > 0:
        #     print(torch.mean(torch.stack(accs)) * 100.0)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    if True and sw is not None and sw.save_this:

        for n in range(N):
            if visibles[0,0,n] > 0:
                sw.summ_traj2ds_on_rgbs('outputs/kp%d_trajs_e_on_rgbs' % n, trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:S], cmap='spring', linewidth=2)
        
        #prep_rgbs = prep_rgbs.unsqueeze(0)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        label_colors = utils.improc.get_n_colors(N) 
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
    
    return metrics

def run_resnet(resnet, d, sw):
    import copy
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    patch_size = 8

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    # H_, W_ = 496, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    segs = (segs > 0).float()
    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run pretrained resnet to get featmaps
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1,2,0).detach().cpu().numpy(), scale_size=[320])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0) # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0 
        featmaps = []
        while idx < S:
            end_id = min(S, idx+bs)
            featmaps_cur = resnet(prep_rgbs[idx:end_id].cuda()) # S, C, h, w
            _, _, h, w = featmaps_cur.shape
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0) # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    first_seg = torch.zeros((1, N, H_//patch_size, W_//patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h*w) # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h*w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

         # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = visibles[0,cnt,n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / frame_tar_avg[n][nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
            else:
                # stay where it was
                # coord_e = trajs_g[0,0,n]
                coord_e = trajs_e[0,cnt-1,n]
            trajs_e[0, cnt, n] = coord_e
            if vis > 0:
                coord_g = trajs_g[0,cnt,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,cnt])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

        if len(accs) > 0:
            print(torch.mean(torch.stack(accs)) * 100.0)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    trajs_e[0,0] = trajs_g[0,0]
    if True and sw is not None and sw.save_this:
        #prep_rgbs = prep_rgbs.unsqueeze(0)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        label_colors = utils.improc.get_n_colors(N) 
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
    
    return metrics

def get_feats_from_timecycle(model, frame):
    with torch.no_grad():
        feat = model.encoderVideo(frame.unsqueeze(0))
    feat_relu = model.relu(feat)
    feat_norm = F.normalize(feat_relu, p=2, dim=1)
    import ipdb; ipdb.set_trace()
    return feat_norm

def run_timecycle(timecycle, d, sw):
    import copy
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    patch_size = 8

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    # H_, W_ = 496, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    segs = (segs > 0).float()
    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run pretrained resnet to get featmaps
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1,2,0).detach().cpu().numpy(), scale_size=[320])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0) # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0 
        featmaps = []
        while idx < S:
            end_id = min(S, idx+bs)
            featmaps_cur = get_feats_from_timecycle(timecycle, prep_rgbs[idx:end_id].cuda()) # S, C, h, w
            _, _, h, w = featmaps_cur.shape
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0) # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    first_seg = torch.zeros((1, N, H_//patch_size, W_//patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h*w) # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h*w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

         # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = visibles[0,cnt,n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / frame_tar_avg[n][nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
            else:
                # stay where it was
                # coord_e = trajs_g[0,0,n]
                coord_e = trajs_e[0,cnt-1,n]
            trajs_e[0, cnt, n] = coord_e
            if vis > 0:
                coord_g = trajs_g[0,cnt,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                # print_('dist', dist)
                area = torch.sum(segs[0,cnt])
                # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

        if len(accs) > 0:
            print(torch.mean(torch.stack(accs)) * 100.0)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    trajs_e[0,0] = trajs_g[0,0]
    if True and sw is not None and sw.save_this:
        #prep_rgbs = prep_rgbs.unsqueeze(0)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        label_colors = utils.improc.get_n_colors(N) 
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
    
    return metrics

def run_timecycle2(model, d, sw):
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 320
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    segs = (segs > 0).float()

    # TODO: normalize rgb
    mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,1,3,1,1).float().cuda()
    std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,1,3,1,1).float().cuda()
    rgbs = ((rgbs / 255.0) - mean) / std

    '''
    imgs_total --- B, S, C, H, W
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    '''
    finput_num_ori = 8
    finput_num = finput_num_ori
    now_batch_size = 4
    cropSize = 320
    cropSize2 = 80
    topk_vis = 5

    corrfeat2_set = []

    imgs_tensor = torch.zeros(now_batch_size, finput_num, 3, H_, W_).cuda()
    target_tensor = torch.zeros(now_batch_size, 1, 3, H_, W_).cuda()
    for itr in range(0, S, now_batch_size):
        startid = itr
        endid = itr + now_batch_size

        if endid > S-finput_num_ori:
            endid = S-finput_num_ori

        now_batch_size2 = endid - startid

        for i in range(now_batch_size2):
            imgs = rgbs[:, itr+i+1:itr+i+finput_num_ori]
            imgs2 = rgbs[:, 0:1]

            imgs = torch.cat((imgs2, imgs), dim=1)
            imgs_tensor[i] = imgs
            target_tensor[i, 0] = rgbs[0, itr+i+finput_num_ori]

        corrfeat2_now = model(imgs_tensor, target_tensor)
        corrfeat2_now = corrfeat2_now.view(now_batch_size, finput_num_ori, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))

        for i in range(now_batch_size2):
            corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

    # create gt
    xy0 = trajs_g[:, 0, :] # B, N, 2
    lbls_resize2 = np.zeros((S, H_ // 8, W_ // 8, N))
    patch_size = 8
    for n in range(N):
        lbls_resize2[0, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long(), n] = 1

    point_tracks = trajs_g[0, 0] / 8 # (20, 2)
    N_point = point_tracks.shape[0]
    accs = []

    mask = restrict_neighborhood(H_ // 8, W_ // 8).reshape(H_ // 8, W_ // 8, H_ // 8, W_ // 8).cpu() # (h, w, h, w)

    for itr in range(S - finput_num_ori):

        imgs = rgbs[:, itr+1:itr+finput_num_ori]
        imgs2 = rgbs[:, 0:1]
        imgs = torch.cat((imgs2, imgs), dim=1)

        corrfeat2 = corrfeat2_set[itr]
        corrfeat2 = torch.from_numpy(corrfeat2)

        out_frame_num = int(finput_num)
        height_dim = corrfeat2.size(2)
        width_dim = corrfeat2.size(3)

        corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
        corrfeat2 *= mask.unsqueeze(0)
        corrfeat2 = corrfeat2.data.cpu().numpy()

        vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
        vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

        atten1d  = corrfeat2.reshape(corrfeat2.shape[0], height_dim * width_dim, height_dim, width_dim)
        ids = np.argpartition(atten1d, -topk_vis, axis=1)[:, -topk_vis:]

        hid = ids // width_dim
        wid = ids % width_dim

        vis_ids_h = wid.transpose(0, 2, 3, 1)
        vis_ids_w = hid.transpose(0, 2, 3, 1)

        predlbls = np.zeros((height_dim, width_dim, N))

        for t in range(finput_num):
            h, w, k = np.meshgrid(np.arange(height_dim), np.arange(width_dim), np.arange(topk_vis), indexing='ij')
            h, w = h.flatten(), w.flatten()

            hh, ww = vis_ids_h[t].flatten(), vis_ids_w[t].flatten()

            if t == 0:
                lbl = lbls_resize2[0, hh, ww, :]
            else:
                lbl = lbls_resize2[t + itr, hh, ww, :]

            np.add.at(predlbls, (h, w), lbl * corrfeat2[t, ww, hh, h, w][:, None])

        predlbls = predlbls / finput_num

        for t in range(N):
            nowt = t
            predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
            predlbls[:, :, nowt] = predlbls[:, :, nowt] / (1e-8 + predlbls[:, :, nowt].max())

        lbls_resize2[itr + finput_num_ori] = predlbls

    trajs_g[:,:,:,0] *= float(512) / float(320)

    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    trajs_e[0,0] = trajs_g[0,0]

    segs_ = segs.reshape(B*S, 1, H_, W_)
    segs_ = F.interpolate(segs_, (320, 512), mode='nearest')
    segs = segs_.reshape(B, S, 1, 320, 512)
    segs = (segs > 0).float()

    for s in range(1, S):
        for n in range(N):
            pred_seg = torch.from_numpy(lbls_resize2[s, :, :, n]).to(device)
            pred_seg = F.interpolate(pred_seg.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
            pred_seg = norm_mask(pred_seg).squeeze(0)

            nz = torch.nonzero(pred_seg)

            if pred_seg[nz[:,0], nz[:,1]].sum() == 0:
                coord_e = trajs_e[0, s-1, n]
            else:
                coord_e = torch.sum(pred_seg[nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / pred_seg[nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
                # import ipdb; ipdb.set_trace()
                coord_e[0] *= float(512) / float(320)

            trajs_e[0, s, n] = coord_e

            vis = visibles[0,s,n]
            if vis > 0:
                coord_g = trajs_g[0,s,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                area = torch.sum(segs[0,s])
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    return metrics

def run_mast(mast, d, sw):
    import copy
    metrics = {}

    file0 = str(d['file0'])
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    segs = d['segs'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    visibles = d['visibles'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)

    if 'extra_videos' in file0:
        animal = file0.split('/')[-3]
    else:
        animal = file0.split('/')[-2]
    metrics['animal'] = animal

    patch_size = 8

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    segs_ = segs.reshape(B*S, 1, H, W)
    H_, W_ = 320, 512
    # H_, W_ = 496, 768
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    segs_ = F.interpolate(segs_, (H_, W_), mode='nearest')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    segs = segs_.reshape(B, S, 1, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    # H, W = H_, W_
    segs = (segs > 0).float()
    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    first_seg = torch.zeros((1, N, H_, W_)).cuda()
    for n in range(N):
        first_seg[0, n, xy0[0, n, 1].long(), xy0[0, n, 0].long()] = 1
    outputs = [first_seg]

    import torchvision.transforms as transforms
    labs = []
    for s in range(S):
        image = np.float32(rgbs[0, s].permute(1,2,0).cpu().numpy()) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([50,0,0], [50,127,127])(image)
        labs.append(image.unsqueeze(0))
    labs = torch.stack(labs, dim=1).to(device)

    for cnt in range(S-1):
        mem_gap = 2 
        ref = 0 
        if ref == 0:
            ref_index = list(filter(lambda x: x <= cnt, [0, 5])) + list(filter(lambda x:x>0,range(cnt,cnt-mem_gap*3,-mem_gap)))[::-1]
            ref_index = sorted(list(set(ref_index)))
        else:
            raise NotImplementedError

        rgb_0 = [labs[:, ind] for ind in ref_index]
        rgb_1 = labs[:, cnt+1]

        anno_0 = [outputs[ind] for ind in ref_index]

        _, _, h, w = anno_0[0].shape

        max_class = N

        with torch.no_grad():
            _output = mast(rgb_0, anno_0, rgb_1, ref_index, cnt+1)
            _output = F.interpolate(_output, (h,w), mode='bilinear')

            outputs.append(_output)

    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    trajs_e[0,0] = trajs_g[0,0]
    for s in range(1, S):
        for n in range(N):
            pred_seg = outputs[s][:, n]
            #pred_seg = norm_mask(pred_seg)[0]
            pred_seg = pred_seg[0]
            nz = torch.nonzero(pred_seg)

            if pred_seg[nz[:,0], nz[:,1]].sum() == 0:
                coord_e = trajs_e[0, s-1, n]
            else:
                coord_e = torch.sum(pred_seg[nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / pred_seg[nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]

            trajs_e[0, s, n] = coord_e

            vis = visibles[0,s,n]
            if vis > 0:
                coord_g = trajs_g[0,s,n] # 2
                dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
                area = torch.sum(segs[0,s])
                thr = 0.2 * torch.sqrt(area)
                correct = (dist < thr).float()
                accs.append(correct)

    pck = torch.mean(torch.stack(accs)) * 100.0
    metrics['pck'] = pck.item()

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    if True and sw is not None and sw.save_this:
        #prep_rgbs = prep_rgbs.unsqueeze(0)
        sw.summ_rgbs('inputs/rgbs', prep_rgbs.unbind(1))
        sw.summ_oneds('inputs/segs', segs.unbind(1))
        label_colors = utils.improc.get_n_colors(N) 
        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_g[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = kp * visibles[0:1,0].reshape(1, N, 1, 1)
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            # print('rgb', rgb.shape)
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('inputs/kp_vis', kp_vis)
        # sw.summ_traj2ds_on_rgbs('inputs/trajs_g_on_rgbs', trajs_g[0:1], prep_rgbs[0:1], cmap='winter', valids=visibles[0:1])
        # sw.summ_traj2ds_on_rgb('inputs/trajs_g_on_rgb', trajs_g[0:1], prep_rgbs[0:1,0], cmap='winter', valids=visibles[0:1])

        kp_vis = []
        for s in range(S):
            kp = utils.improc.draw_circles_at_xy(trajs_e[0:1,s], H_, W_, sigma=4).squeeze(2) # 1, N, H_, W_
            kp = sw.summ_soft_seg_thr('', kp, label_colors=label_colors, only_return=True).cuda()
            kp_any = (torch.max(kp, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            rgb = (torch.mean(rgbs[:,s], dim=1, keepdim=True).repeat(1, 3, 1, 1)).byte()
            kp[kp_any==0] = rgb[kp_any==0]
            kp_vis.append(kp)
        sw.summ_rgbs('outputs/kp_vis', kp_vis)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_e_on_rgbs', trajs_e[0:1], prep_rgbs[0:1,:S], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')

    return metrics


def train():

    # default coeffs (don't touch)
    init_dir = ''
    coeff_prob = 0.0
    use_augs = False

    # device = 'cpu:0'
    # device = 'cuda'

    # test keypoint propagation on BADJA
    exp_name = 'ba00' # copy from other repo 

    init_dir = 'reference_model'
    
    ## choose hyps
    B = 1
    S = 8
    lr = 1e-4
    grad_acc = 1
    
    stride = 4

    max_iters = 7
    log_freq = 1
    # max_iters = 7
    log_freq = 99999
    
    save_freq = 10000
    shuffle = False
    cache = False
    mini_len = 101
    
    # actual coeffs
    coeff_prob = 1.0
    use_augs = True
    # use_augs = False

    ## autogen a name
    model_name = "%02d_%d" % (B, S)
    model_name += "_%s" % exp_name
    
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_test_on_badja'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    train_dataset = badjadataset.BadjaDataset()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=1)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    # # raft = nets.raftnet.RaftNet(ckpt_name='../RAFT/models/raft-sintel.pth').cuda()
    # raft = Raftnet(ckpt_name='../RAFT/models/raft-things.pth').cuda()
    raft = Raftnet(ckpt_name='../RAFT/models/raft-sintel.pth').cuda()
    raft.eval()

    singlepoint = Singlepoint(stride=stride).cuda()
    if init_dir:
       _ = saverloader.load(init_dir, singlepoint)
    singlepoint.eval()

    # stride = 8
    # crop_size = (368,496)
    # H, W = crop_size
    # H8, W8 = H//8, W//8
    # dof = Dofperionet(H8, W8, S=S, stride=stride).cuda()
    # dof_init_dir = 'checkpoints/08_8_512_1e-4_p1_do28_16:16:10'
    # if dof_init_dir:
    #     _ = saverloader.load(dof_init_dir, dof)
    # dof.eval()
    

    n_pool = 1000
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
    vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_occ_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_inv2inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    flow_pool_t = utils.misc.SimplePool(n_pool, version='np')

    # timecycle
    if False:
        # create timecycle model
        import sys
        sys.path.append("/home/zhaoyuaf/tracking_sol/TimeCycle")
        import models.videos.model_test as video3d
        timecycle = video3d.CycleTime(class_num=49, trans_param_num=3, pretrained=False, temporal_out=4)
        timecycle.to(device)

        # load from checkpoint
        checkpoint = torch.load("/home/zhaoyuaf/tracking_sol/TimeCycle/checkpoint_14.pth.tar")
        pretrained_dict = checkpoint['state_dict']
        model_dict = timecycle.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        timecycle.load_state_dict(model_dict)
        del checkpoint

    # dino
    if True:
        patch_size = 8
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size)
        for p in dino.parameters():
            p.requires_grad = False
        dino.eval()
        dino.to(device)

    # resnet
    if False:
        from torchvision import models
        resnet = models.resnet50(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-4])
        resnet.eval()
        resnet.to(device)

    if False:
        import sys
        sys.path.append("/home/yunchuz/andy/MAST")
        from models.mast import MAST

        parser = argparse.ArgumentParser(description='MAST')
        parser.add_argument('--ref', type=int, default=0)
        args = parser.parse_args()

        args.training = False
        mast = MAST(args)

        checkpoint = torch.load("/home/yunchuz/andy/MAST/checkpoint.pt")
        mast.load_state_dict(checkpoint['state_dict'])
        mast.eval()
        mast.to(device)
    else:
        mast = None
        
    results = []
    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=24,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        if cache:
            if len(samples) < mini_len:
                sample = next(train_iterloader)
                samples.append(sample)
            else:
                sample = samples[global_step % mini_len]
        else:
            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        with torch.no_grad():
            # metrics = run_static(sample, sw_t)
            metrics = run_singlepoint(singlepoint, sample, sw_t)
            # metrics = run_raft(raft, sample, sw_t)
            # metrics = run_raft_skip(raft, sample, sw_t)
            # metrics = run_timecycle(timecycle, sample, sw_t)
            # metrics = run_timecycle2(timecycle, sample, sw_t)
            # metrics = run_dino(dino, sample, sw_t)
            # metrics = run_resnet(resnet, sample, sw_t)
            # metrics = run_mast(mast, sample, sw_t)
        
        # results.append('%.1f' % (metrics['pck']))
        # results.append('%.1f \t %s' % (metrics['pck'], metrics['animal']))
        results.append(metrics['pck'])

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; %s; pck %.1f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            metrics['animal'], metrics['pck']))

        rp = []
        for result in results:
            rp.append('%.1f' % (result))
        rp.append('avg %.1f' % (np.mean(results)))
        print('results', rp)

    # print('results', results)
    # for result in results:
    #     print(result)
    # print(result for result in results)

    # rp = []
    # for result in results:
    #     rp.append('%.1f' % (result))
    # rp.append('avg %.1f' % (np.mean(results)))
    # print('results', rp)
        
            
    writer_t.close()
    
if __name__ == '__main__':
    train()
    



