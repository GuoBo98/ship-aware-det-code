import torch.nn as nn

class ResNet(nn.module):
    def _init_(self,x):
        '''定义自己的Batch-Norm'''
        self.norm1 = nn.BatchNorm2d(64,affine=True)
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
        return tuple(outs)