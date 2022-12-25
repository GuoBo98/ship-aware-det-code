import math
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init


class PcbFuse(nn.Module):
    '''PCB切片然后对特征加权'''

    def __init__(self,
                 fuse_mode = 0):

        super(PcbFuse, self).__init__()
        
        weight_length = 3
        self.weight = nn.Parameter(torch.ones(weight_length))
        self.weight_path = weight_log_path = '/home/guobo/OBBDetection/_weight_log/'
        self.name = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
        self.count = 0
        self.epoch = 1

    def forward(self, x_input):
        weight = F.softmax(self.weight, 0)
        part1 = x_input[:,:,:,0:4] * weight[0]
        part2 = x_input[:,:,:,4:8] * weight[1]
        part3 = x_input[:,:,:,8:12] * weight[2]
        x_output = torch.cat((part1,part2,part3),3)
        
        self.count = self.count + 1 
        if self.count % 667 == 0:
            self.epoch = self.epoch + 1
        if self.count % 50 == 0:
            with open(os.path.join(self.weight_path,str(self.name) + '.txt'),'a') as f:
                # f.write('Switch-Time'.ljust(20) + str(self.switch_time) + ' ' + str(time.asctime()) + '\n')
                f.write( str('epoch_' + str(self.epoch)).ljust(10) + str(weight).ljust(3) + '\n' ) 

        return x_output
