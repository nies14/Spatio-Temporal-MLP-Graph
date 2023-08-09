from __future__ import absolute_import, division

import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedJacobiConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True,):
        super(WeightedJacobiConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.zeros(size=(1, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(1,adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj_1 = adj # one_hop neighbors
        self.adj2_1 = nn.Parameter(torch.ones_like(self.adj_1))
        nn.init.constant_(self.adj2_1, 1e-6)

        self.alpha=0.1#0.5#0.2
        self.lamda=0.5#1.0#0.5
        #l=4#####
        self.beta = .7 #math.log(self.lamda/self.l+1.0)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input1, input2):

        #I = torch.eye(self.adj_1.size(1), dtype=torch.float).to(input1.device)
        h1 = torch.matmul(input1,self.W[0])
        h2 = torch.matmul(input1,self.W[1])
        #a = torch.einsum('njc,jj->njc', (h2,I))
        b = self.M[0]*h1

        adj =  self.adj_1.to(input1.device) + self.adj2_1.to(input1.device)
        A_0 = (adj.T + adj)/2
        c = torch.einsum('njc,jj->njc', (((1-self.alpha)*self.M[0])*h1), A_0)  #torch.matmul(A_0,((1-self.alpha)*self.M[0])*h1)
        #d = torch.einsum('njc,jj->njc', (((1-self.alpha)*self.M[0])*h2), A_0 * (1-I))

        x1 = torch.matmul(input2,self.W2[0])
        e = self.alpha * self.M[0]*x1

        #output = a - b + c + d + e
        output = h2 - b + c + e

        if self.bias is not None:
            return output + self.bias.view(1,1,-1), x1
        else:
            return output, x1


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
