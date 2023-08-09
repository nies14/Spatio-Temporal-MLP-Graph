import argparse
import os
import math



class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # ===============================================================
        #                     Dataset options
        # ===============================================================
        self.parser.add_argument('--dataset', type=str, default='h36m', help='dataset')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME',
                            help='2D detections to use {gt||cpn_ft_h36m_dbb}')
        self.parser.add_argument('--data_augmentation', type=bool, default=True,help='disable train-time flipping')##
        self.parser.add_argument('--test_augmentation', type=bool, default=True,help='flip and fuse the output result')
        #self.parser.add_argument('--data_augmentation', type=bool, default=False, help='disable train-time flipping')##
        #self.parser.add_argument('--test_augmentation', type=bool, default=False, help='flip and fuse the output result')
        
        self.parser.add_argument('--crop_uv', type=int, default=0,help='if crop_uv to center and do normalization')

        self.parser.add_argument('--root_path', type=str, default='./dataset/', help='dataset root path')
        self.parser.add_argument('--cal_uvd', type=bool, default=True, help='calculate uvd error as well')
        self.parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                            help='actions to train/test on, separated by comma, or * for all')
        self.parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                            help='downsample frame rate by factor (semi-supervised)')
        self.parser.add_argument('--subset', default=1, type=float, metavar='FRACTION',
                            help='reduce dataset size by fraction')
        self.parser.add_argument('-s', '--stride', default=1, type=int, metavar='N',
                            help='chunk size to use during training')
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False,help='if reverse the video to augment data')    
        

        # ===============================================================
        #modification for layer number, dropout and hid_dim 
        self.parser.add_argument('-l', '--num_layers', default=3, type=int, metavar='N', help='num of residual layers')
        self.parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
        self.parser.add_argument('--mlpdropout', default=0.0, type=float, help='mlp dropout rate')
        self.parser.add_argument('-z', '--hid_dim', default=384, type=int, metavar='N', help='num of hidden dimensions')
        self.parser.add_argument('-ds', '--spatial_hid_dim', default=384, type=int, metavar='N', help='num of hidden dimensions')
        self.parser.add_argument('-dc', '--channel_hid_dim', default=384, type=int, metavar='N', help='num of hidden dimensions')
        self.parser.add_argument('-norm', '--norm', default=0.01, type=float, metavar='N', help='constraint  of sparsity')
        self.parser.add_argument('--save_dir', default='', type=str, help='model save dir')
        #end modification 
        # ===============================================================

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--pro_train', type=int, default=0,help='if start train process')
        self.parser.add_argument('--pro_test', type=int, default=1,help='if start test process')
        self.parser.add_argument('--nepoch', type=int, default=31, help='number of epochs')#
        self.parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--learning_rate', type=float, default=5e-3)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=4,help='give a large lr decay after how manys epochs')
        self.parser.add_argument('--sym_penalty', type=int, default=0, help='if add sym penalty add on train_multi')
        self.parser.add_argument('--co_diff', type=float, default=0)
        self.parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')


        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer for SGD')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
        self.parser.add_argument('-lrd', '--lr_decay', default=0.90, type=float, metavar='LR',
                            help='learning rate decay per epoch')#
        self.parser.add_argument('--save_skl', type=bool, default=False)#

        # ===============================================================
        #                     Model options
        self.parser.add_argument('--pad', default=0, type=int)
        self.parser.add_argument('--show_protocol2', action='store_true')#
        self.parser.add_argument('--model_doc', type=str, default='modulated_gcn', help='current model document name')
        self.parser.add_argument('--layout', type=str, default='hm36_gt', help='dataset used')
        self.parser.add_argument('--strategy', type=str, default='spatial', help='dataset used')
        self.parser.add_argument('--save_model', type=int, default=0, help='if save model')
        self.parser.add_argument('--save_out_type', type=str, default='xyz', help='xyz/uvd/post/time')

        self.parser.add_argument('--post_refine', action='store_true', help='if use post_refine model')
        self.parser.add_argument('--wj_gcn_reload', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--post_refine_reload', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--previous_dir', type=str,
                                 default='./ckpt/module_256/',
                                 help='previous output folder')
        self.parser.add_argument('--wj_gcn_model', type=str, default='model_module_gcn_7_eva_post_4939.pth', help='model name')
        self.parser.add_argument('--post_refine_model', type=str, default='model_post_refine_7_eva_post_4939.pth',
                                 help='model name')

        self.parser.add_argument('--n_joints', type=int, default=16, help='number of joints, 16 for human body 21 for hand pose')
        self.parser.add_argument('--out_joints', type=int, default=16, help='number of joints, 16 for human body 21 for hand pose')
        self.parser.add_argument('--out_all', type=bool, default=True, help='output 1 frame or all frames')
        self.parser.add_argument('--in_channels', type=int, default=2, help='expected input channels here 2')
        self.parser.add_argument('--out_channels', type=int, default=3, help='expected input channels here 2')
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf,
                            help='threshold data:reg_RGB_3D/reg_3D_3D')
        self.parser.add_argument('-previous_wj_gcn_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('-previous_post_refine_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('--lamda', '--weight_L1_norm', default=0.1, type=float, metavar='N', help='scale of L1 Norm')
        self.parser.add_argument('--manual_seed','--manual_seed', default=1420, type=int, help='manual seed value')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

        '''
        self.opt.save_dir = './results/'+'%d_frame/'%(self.opt.pad*2+1) +self.opt.model_doc + '/'+ \
        '%spose_refine/'%('' if self.opt.post_refine else 'no_')
        '''

        if self.opt.dataset == 'h36m':
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'
            #self.opt.subjects_test = 'S1,S5,S6,S7,S8'



        if self.opt.keypoints == 'cpn_ft_h36m_dbb':
            #self.opt.save_dir += 'cpn/'
            self.opt.layout = 'hm36_gt'

        elif self.opt.keypoints == 'gt':
            #self.opt.save_dir += 'gt/'
            self.opt.layout = 'hm36_gt'


        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')


        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        print(self.opt)
        return self.opt
