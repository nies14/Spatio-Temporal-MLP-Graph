from einops import rearrange
from models.wj_conv import WeightedJacobiConv
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        
        return self.net(x)
    
class _GraphConvI(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):

        super(_GraphConvI, self).__init__()

        self.graphConv = WeightedJacobiConv(input_dim, output_dim, adj)
        self.norm1 = nn.BatchNorm1d(output_dim)
        if p_dropout is not None:
            self.dropout1 = nn.Dropout(p_dropout)
        else:
            self.dropout1 = None
        
        self.act = nn.GELU()
        self.graphConvII = WeightedJacobiConv(output_dim, input_dim, adj)
        self.norm2 = nn.BatchNorm1d(input_dim)
        if p_dropout is not None:
            self.dropout2 = nn.Dropout(p_dropout)
        else:
            self.dropout2 = None

    def forward(self, x, initial):

        input = x
        x, initial = self.graphConv(x, initial)
        x = x.transpose(1, 2)
        x = self.norm1(x).transpose(1, 2)
        if self.dropout1 is not None:
            x = self.dropout1(self.act(x))
        else:
            x = self.act(x)

        x, _ = self.graphConvII(x, initial)
        x = x.transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        if self.dropout2 is not None:
            x = self.dropout2(self.act(x))
        else:
            x = self.act(x)

        return x

class MixerBlock(nn.Module):

    def __init__(self, spatial_dim, channel_dim, adj = None, dropout = 0., opt=None):

        super().__init__()

        self.spatial_mix = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            Rearrange('b n d -> b d n'),
            FeedForward(opt.n_joints, spatial_dim, opt.mlpdropout),
            Rearrange('b d n -> b n d')
        )
        
        self.gconv = _GraphConvI(adj, spatial_dim, channel_dim, opt.dropout)

    def forward(self, x, initial):

        x = x + self.spatial_mix(x)     
        x = x + self.gconv(x, initial)

        return x

class GraphMlpMixer(nn.Module):

    def __init__(self, adj, in_channels, spatial_dim, channel_dim, depth, p_dropout = 0.2, opt = None):

        super().__init__()
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(spatial_dim, channel_dim, adj, opt = opt))

        self.layer_norm = nn.LayerNorm(spatial_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(spatial_dim, 3)
        )

    def forward(self, x, initial):

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x, initial)

        x = self.layer_norm(x)

        return self.mlp_head(x)

class STMLPGraph(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None, opt = None):
        super(STMLPGraph, self).__init__()

        frame = (opt.pad*2) + 1

        self.to_patch_embedding = nn.Linear(frame * 2, opt.hid_dim)
        
        self.gconv_layers = GraphMlpMixer(adj, in_channels = opt.hid_dim, spatial_dim = opt.spatial_hid_dim, channel_dim = opt.channel_hid_dim, depth=opt.num_layers, p_dropout=p_dropout, opt=opt)
        
    def forward_spatial_temporal(self,x):

        x = x.squeeze(4)
        x = rearrange(x, 'b c f j -> b j (f c)')

        out = self.to_patch_embedding(x)
        initial = out
        out = self.gconv_layers(out, initial)
        out = out.permute(0,2,1)
        out = out.unsqueeze(2)
        out = out.unsqueeze(4)
        return out

        
    def forward(self, x):    

        out = self.forward_spatial_temporal(x)
        return out