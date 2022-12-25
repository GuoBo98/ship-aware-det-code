from time import sleep
from turtle import forward
import torch
import torch.nn as nn
from ...mmdet.models.roi_heads.bbox_heads.obb.obbox_head import OBBoxHead

class ResNet(nn.Module):
    def _init_(self,x):
        '''定义自己的Batch-Norm'''
        self.norm1 = nn.BatchNorm2d(64,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,stride=2)
        '''super的意义是首先找到ResNet的父类,这里是nn.Module,然后将类Net的对象self转化成父类nn.Module的对象。然后调用__init()__对自己进行初始		   化，这是对继承自父类的属性进行初始化'''
        super(ResNet,self).__init__()
        
    def forward(self,x):
        ''' x.shape is 2*3*1024*1024 N*C*H*W'''
        x = self.conv1(x)
        ''' self.conv1 is Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        	so, x.shape is (W + 2* padding - kernel_size)/stride + 1 = [2, 64, 512, 512]'''
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        '''resnet有四个stage,通过ResNet的1*1卷积和stride=2实现每个stage特征图大小除以2
           outs.shape: torch.Size([2, 256, 256, 256],[2, 512, 128, 128],[2, 1024, 64, 64],[2, 2048, 32, 32])'''
        outs = []
        for i , layername in enumerate(self.reslayers):
            res_layer = getattr(self,layername)
            x = res_layer(x)
            outs.append(x)
        
class FPN(nn.Module):
    '''in_channels = [256,512,1024,2048] , out_channels = 256'''
    def __init__(self,In_channels,Out_channels):
        
        super(FPN,self).__init__()
        
        '''定义横向连接的卷积和最后fpn的3*3的卷积'''
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(len(In_channels)):
            l_conv = nn.Conv2d(in_channels=In_channels[i] , out_channels= Out_channels, kernel_size= 1)
            fpn_conv = nn.Conv2d(in_channels = Out_channels , out_channels= Out_channels , kernel_size= 3 , padding= 1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        
    
    def forward(self,inputs):
        
        start_level = 0
        '''建立横向连接部分,统一通道数'''
        laterals = [lateral_conv(inputs[i + start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        '''建立自顶向下部分'''
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1, 0, -1 ):
            pre_shape = laterals[i-1].shape[2:]
            laterals[i - 1] += nn.functional.interpolate(laterals[i],size=pre_shape,mode='nearest')
        '''通过3*3的卷积'''
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        '''在检测中一般要把P5再下采样一个P6出来'''
        outs.append(nn.functional.max_pool2d(outs[-1],1,stride=2))
        
        return tuple(outs)

'''FPN-Test
a = FPN(In_channels=[256,512,1024,2048] , Out_channels=256)

inputs = []
C2 = torch.randn(size = (1,256,256,256))
C3 = torch.randn(size = (1,512,128,128))
C4 = torch.randn(size = (1,1024,64,64))
C5 = torch.randn(size = (1,2048,32,32))

inputs.append(C2)
inputs.append(C3)
inputs.append(C4)
inputs.append(C5)

inputs = tuple(inputs)

outputs = a(inputs)
'''

class OBBConvFCBBoxHead(nn.Module):
    def __init__(self):
        
        self.shared_fc1 = nn.Linear(in_features=12544, out_features=1024,bias=True)
        self.shared_fc2 = nn.Linear(in_features=1024,out_features=1024,bias = True)
        self.fc_bn_1 = nn.BatchNorm1d(1024,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.fc_bn_2 = nn.BatchNorm1d(1024,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        super().__init__()
    
    def forward(self,x):
        x = x.flatten(1)
        x = self.relu()
        
        
        
   
    

            
        
        
            
            
        
        
        
        
        
        

        