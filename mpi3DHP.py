from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import random
import argparse
import numpy as np
import os.path as path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.common.graph_utils import adj_mx_from_skeleton
from data.common.PoseGenerator import PoseBuffer
from opt1 import opts
from utils.data_utils import unNormalizeData as unData
from data.common.camera import *
from data.common.utils import AverageMeter
from models.STMLPGraph import STMLPGraph

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

def evaluate(data_loader, model_pos_eval, device, summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_auc = AverageMeter()
    epoch_pck = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()

    for i, temp in enumerate(data_loader):
        targets_3d, inputs_2d = temp[0], temp[1]

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = inputs_2d.to(device)

        with torch.no_grad():
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[:, joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(inputs_2d_flip.permute(0,2,1)).squeeze().permute(0,2,1).cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d.permute(0,2,1)).squeeze().permute(0,2,1).cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:
                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()

        # caculate the relative position.
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint

        # compute AUC and PCK
        pck = compute_PCK(targets_3d.cpu().numpy(), outputs_3d.cpu().numpy())
        epoch_pck.update(pck, num_poses)
        auc = compute_AUC(targets_3d.cpu().numpy(), outputs_3d.cpu().numpy())
        epoch_auc.update(auc, num_poses)

        #print("PCK: ",pck," AUC: ",auc," \n")

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if writer:
        #writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p1score' + tag, epoch_p1.avg, summary.epoch)
        #writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p2score' + tag, epoch_p2.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_pck' + tag, epoch_pck.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_auc' + tag, epoch_auc.avg, summary.epoch)

    print("PCK: ", epoch_auc.avg)
    print("AUC: ", epoch_pck.avg)
    return epoch_pck.avg, epoch_auc.avg

def compute_PCK(gts, preds, scales=1000, eval_joints=None, threshold=150):
    PCK_THRESHOLD = threshold
    sample_num = len(gts)
    total = 0
    true_positive = 0
    if eval_joints is None:
        eval_joints = list(range(gts.shape[1]))

    for n in range(sample_num):
        gt = gts[n]
        pred = preds[n]
        # scale = scales[n]
        scale = 1000
        per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
        true_positive += (per_joint_error < PCK_THRESHOLD).sum()
        total += per_joint_error.size

    pck = float(true_positive / total) * 100
    return pck


def compute_AUC(gts, preds, scales=1000, eval_joints=None):
    # This range of thresholds mimics 'mpii_compute_3d_pck.m', which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = np.linspace(0, 150, 31)
    pck_list = []
    for threshold in thresholds:
        pck_list.append(compute_PCK(gts, preds, scales, eval_joints, threshold))

    auc = np.mean(pck_list)

    return auc

def data_preparation(opt):

    file_path = opt.root_path + opt.mpi_3dhp_name
    mpi3d_npz = np.load(file_path)

    input_2D = mpi3d_npz['pose2d']
    output_3d = mpi3d_npz['pose3d']

    mpi3d_loader = DataLoader(PoseBuffer([output_3d], [input_2D]),
                              batch_size=opt.batchSize,
                              shuffle=False, num_workers=6, pin_memory=True)


    return mpi3d_loader
    

def main(args):
    mpi3dhp_loader = data_preparation(args)
    device = torch.device("cuda")
    
    root_path = opt.root_path
    #dataset_path = path.join('./dataset/data_3d_h36m.npz')
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    
    if args.dataset == 'h36m':
        from data.common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path,opt=args)
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = STMLPGraph(adj, hid_dim = opt.num_layers, p_dropout=p_dropout, nodes_group=None, opt = opt).to(device)
    
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    #ckpt_path = './results/New/GT/Pose-Refine/128_128_L_3_frame_1/wo_pose_refine/model_wj_gcn_6_eva_xyz_3634.pth'
    ckpt_path = opt.previous_dir + opt.wj_gcn_model

    pre_dict_module_gcn = torch.load(os.path.join(ckpt_path))
    module_gcn_dict = model_pos.state_dict()
    if path.isfile(ckpt_path):
        for name, key in module_gcn_dict.items(): 
            if name.startswith('A') == False:
                module_gcn_dict[name] = pre_dict_module_gcn[name]
                #module_gcn_dict[name]=pre_dict_module_gcn['state_dict'][name]
    
        model_pos.load_state_dict(module_gcn_dict)

    a,b = evaluate(mpi3dhp_loader, model_pos, device, flipaug='flipaug')
    
if __name__ == '__main__':
    opt = opts().parse()
    main(opt) # import opt)
