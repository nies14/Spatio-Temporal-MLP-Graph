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

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    #l2_error = torch.mean(torch.norm((predicted - target), dim=len(target.shape) - 1), -1).squeeze()
    #print('each joint error:', torch.norm((predicted - target), dim=len(target.shape) - 1))
    #index = np.where(l2_error.cpu().detach().numpy() > 0.3)  # mean body l2 distance larger than 300mm
    #value = l2_error[l2_error > 0.3]
    #print('Index of mean body l2 distance larger than 300mm', index, value)
    return torch.mean(torch.norm((predicted - target), dim=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

def evaluate(data_loader, model_pos_eval, device, summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

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
                outputs_3d_flip = model_pos_eval(inputs_2d_flip).squeeze().cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d).squeeze().cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:
                outputs_3d = model_pos_eval(inputs_2d).squeeze().cpu()

        # caculate the relative position.
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print("MPJPE: ", epoch_loss_3d_pos.avg)
    print("P-MPJPE: ", epoch_loss_3d_pos_procrustes.avg)
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg

def data_preparation():
    #difficult_pose_data = np.load('./dataset/whole_body_S0.05_f5_1_gt.npz')
    difficult_pose_data = np.load(opt.difficult_pose_data + opt.difficult_pose_data_name)
    difficult_pose_data_loader = DataLoader(PoseBuffer([difficult_pose_data['pose_3d'][0]]
                                                     , [difficult_pose_data['pose_2d'][0]])
                                                     , batch_size=256
                                                     , shuffle=False
                                                     , num_workers=6
                                                     , pin_memory=True)

    return difficult_pose_data_loader

def main(opt):
    mpi3d_loader = data_preparation()
    device = torch.device("cuda")
    
    root_path = opt.root_path
    #dataset_path = path.join('./dataset/data_3d_h36m.npz')
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    
    if opt.dataset == 'h36m':
        from data.common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path, opt)
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    stride = opt.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    p_dropout = (None if opt.dropout == 0.0 else opt.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = STMLPGraph(adj, hid_dim = opt.num_layers, p_dropout=p_dropout, nodes_group=None, opt = opt).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    #ckpt_path = './results/New/GT/Pose-Refine/128_128_L_3_frame_1/wo_pose_refine/model_wj_gcn_6_eva_xyz_3634.pth'
    ckpt_path = opt.previous_dir + opt.mlp_graph_model
    
    pre_dict_module_gcn = torch.load(os.path.join(ckpt_path))
    module_gcn_dict = model_pos.state_dict()
    if path.isfile(ckpt_path):
        for name, key in module_gcn_dict.items(): 
            if name.startswith('A') == False:
                module_gcn_dict[name] = pre_dict_module_gcn[name]
                #module_gcn_dict[name] = pre_dict_module_gcn['state_dict'][name]
    
        model_pos.load_state_dict(module_gcn_dict)

    a,b = evaluate(mpi3d_loader, model_pos, device, flipaug='flip')

if __name__ == '__main__':
    opt = opts().parse()
    main(opt) # import opt)


