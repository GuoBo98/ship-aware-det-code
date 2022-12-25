import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init


class PcbFuse(nn.Module):
    '''PCB切片然后对特征加权'''

    def __init__(self):

        super(PcbFuse, self).__init__()
        
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        self.fuse_weight_1.data.fill_(0.33)
        self.fuse_weight_2.data.fill_(0.33)
        self.fuse_weight_3.data.fill_(0.33)

    def forward(self, x_input):
        part1 = x_input[:,:,0:4,:] * self.fuse_weight_1
        part2 = x_input[:,:,4:8,:] * self.fuse_weight_2
        part2 = x_input[:,:,8:,:] * self.fuse_weight_3
        
        return x_input
