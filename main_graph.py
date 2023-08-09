from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from log import Logger
import math
import os
from opt1 import opts
import random
import time
import logging
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import sys
import time
import h5py
import copy
import re

import numpy as np
import socket

from utils.data_utils import define_actions
from utils.utils1 import save_model
import torch.optim as optim
from nets.post_refine import post_refine
from train_graph_time import train, val

# modification #
from data.common.data_utils import read_3d_data
from data.common.graph_utils import adj_mx_from_skeleton
from models.STMLPGraph import STMLPGraph

model = {} # model list 
opt = opts().parse() # import args


from data.load_data_hm36 import Fusion # data fusion to prepare data 



def seed(value=1):
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(value)
    random.seed(value)

seed(opt.manual_seed)

try:
    os.makedirs(opt.save_dir)
except OSError:
    pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(opt.save_dir, 'train_test.log'), level=logging.INFO)
logging.info('======================================================')

# load data
root_path = opt.root_path
if opt.dataset == 'h36m':
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    from data.common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path, opt)


else:
    raise KeyError('Invalid dataset')


actions = define_actions(opt.actions)


lr = opt.learning_rate
p_dropout = (None if opt.dropout == 0.0 else opt.dropout)
adj = adj_mx_from_skeleton(dataset.skeleton())

# load model
model['wj_gcn'] = STMLPGraph(adj, opt.hid_dim, num_layers=opt.num_layers, p_dropout=p_dropout, nodes_group=None, opt = opt).cuda()
model['post_refine'] = post_refine(opt).cuda()



if opt.pro_train:
    train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=False)
if opt.pro_test:
    test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

#3. set optimizer
total_param=0
all_param = []

all_param += list(model['wj_gcn'].parameters())
total_param += sum(p.numel() for p in model['wj_gcn'].parameters())

if opt.post_refine:
    all_param += list(model['post_refine'].parameters())
    total_param += sum(p.numel() for p in model['post_refine'].parameters())


if opt.optimizer == 'SGD':
    optimizer_all = optim.SGD(all_param, lr=opt.learning_rate, momentum=0.9, nesterov=True, weight_decay=opt.weight_decay)
elif opt.optimizer == 'Adam':
    optimizer_all = optim.Adam(all_param, lr=lr, amsgrad=True)
optimizer_all_scheduler = optim.lr_scheduler.StepLR(optimizer_all, step_size=5, gamma=0.1)

print("==> Total parameters: {:.2f}M".format(total_param / 1000000.0))

#4. Reload model

wj_gcn_dict = model['wj_gcn'].state_dict()

if opt.wj_gcn_reload == 1: 
    
    pre_dict_wj_gcn = torch.load(os.path.join(opt.previous_dir, opt.wj_gcn_model))
    #for name, key in stgcn_dict.items():
    for name, key in wj_gcn_dict.items(): 
        if name.startswith('A') == False:
           
            wj_gcn_dict[name] = pre_dict_wj_gcn[name]
    
    model['wj_gcn'].load_state_dict(wj_gcn_dict)

post_refine_dict = model['post_refine'].state_dict()
if opt.post_refine_reload == 1:
    pre_dict_post_refine = torch.load(os.path.join(opt.previous_dir, opt.post_refine_model))
    for name, key in post_refine_dict.items():
        post_refine_dict[name] = pre_dict_post_refine[name]
    model['post_refine'].load_state_dict(post_refine_dict)



#5.Set criterion
criterion = {}
criterion['MSE'] = nn.MSELoss(size_average=True).cuda()
criterion['L1'] = nn.L1Loss(size_average=True).cuda()


logger = Logger(os.path.join(opt.save_dir, 'log.txt'))
logger.set_names(['epoch', 'error_eval_p1', 'error_eval_p2'])

#training process
for epoch in range(1, opt.nepoch):
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
    torch.cuda.synchronize()
    # switch to train
    if opt.pro_train == 1:
        timer = time.time()
        print('======>>>>> training <<<<<======')
        print('frame_number: %d' %(opt.pad*2+1))
        print('processing file %s:' %opt.model_doc)
        print('learning rate %f' % (lr))
        mean_error = train(opt, actions, train_dataloader, model, criterion, optimizer_all)
        timer = time.time() - timer
        timer = timer / len(train_data)
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

    # switch to test
    if opt.pro_test == 1:
        with torch.no_grad():
            timer = time.time()
            print('======>>>>> test<<<<<======')
            print('frame_number: %d' %(opt.pad*2+1))
            print('processing file %s:' %opt.model_doc)
            mean_error = val(opt, actions, test_dataloader, model, criterion,epoch,logger)
            timer = time.time() - timer
            timer = timer / len(test_data)
            print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

            if opt.save_out_type == 'xyz':
                data_threshold = mean_error['xyz']

            elif opt.save_out_type == 'post':
                data_threshold = mean_error['post']


            if opt.save_model and data_threshold < opt.previous_best_threshold:
                opt.previous_wj_gcn_name = save_model(opt.previous_wj_gcn_name, opt.save_dir, epoch, opt.save_out_type, data_threshold, model['wj_gcn'], 'wj_gcn')

                if opt.post_refine:
                    opt.previous_post_refine_name = save_model(opt.previous_post_refine_name, opt.save_dir, epoch, opt.save_out_type,
                                                        data_threshold, model['post_refine'], 'post_refine')
                #print("data_threshold: ",data_threshold)
                opt.previous_best_threshold = data_threshold


    if epoch % opt.large_decay_epoch == 0:
        for param_group in optimizer_all.param_groups:
            param_group['lr'] *= opt.lr_decay
            lr *= opt.lr_decay