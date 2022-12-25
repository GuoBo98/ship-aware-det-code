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
        
        weight_length = 3
        self.weight = nn.Parameter(torch.ones(weight_length))

    def forward(self, x_input):
        
        weight = F.softmax(self.weight, 0)
               
        return x_input
