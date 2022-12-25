import torch
import torch.nn as nn 

class FPN(nn.Module):
    def __init__(self,Inchannels,Outchannels):
        super().__init__()
        self.laterals = nn.ModuleList()
        self.fpnconvs = nn.ModuleList()
        for i in range(len(Inchannels)):
            lateral_conv = nn.Conv2d(in_channels=Inchannels[i],out_channels=Outchannels,kernel_size=1)
            fpn_conv = nn.Conv2d(in_channels=Outchannels,out_channels=Outchannels,kernel_size=3,padding=1)
            self.laterals.append(lateral_conv)
            self.fpnconvs.append(fpn_conv)
    
    def forwad(self,inputs):
        '''第一个横向连接统一通道数'''
        _laterals = [lateral(inputs[i]) for i,lateral in enumerate(self.laterals)]
        '''自顶向下加权'''
        used_stage = len(inputs)
        for i in range(used_stage-1,0,-1):
            prev =  _laterals[i-1].shape[2:]
            _laterals[i-1] += nn.functional.interpolate(_laterals[i],size =  prev,mode = 'nearest')
            outs = []
        '''过3*3卷积再提取下特征'''
        for i in range(used_stage):
            outs.append(self.fpnconvs[i](_laterals[i]))
        '''检测网络再下采样一倍'''
        outs.append(nn.MaxPool2d(outs[-1],1,stride=2))
        return outs

torch.randn(size = (1,256,512,512))

        
norm = nn.BatchNorm2d(64)
input = torch.randn(size = (1,64,512,512))
a = norm(input)
        
layer = nn.LayerNorm(input.shape[1:])   