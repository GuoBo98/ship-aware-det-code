import torch
import torch.nn as nn 

batch_norm1 = nn.BatchNorm2d(64) # NCHW ---> C
input = torch.randn(size = (1,64,512,512))
output = batch_norm1(input)

layer_norm = nn.LayerNorm(input.shape[1:])
output = layer_norm(input[1:])



def iou(box1:torch.Tensor , box2:torch.Tensor):
    
    '''box1 --> [N,4] , box2 --> [M,4]'''
    N = box1.shape[0]
    M = box2.shape[0]
    
    tl = torch.max(box1[:,:2].unsqueeze(1).expand(N,M,2),
                   box2[:,:2].unsqueeze(0).expand(N,M,2))
    
    br = torch.max(box1[:,2:].unsqueeze(1).expand(N,M,2),
                   box2[:,2:].unsqueeze(0).expand(N,M,2))
    
    wh = br - tl 
    wh[wh<0] = 0 
    
    inter = wh[:,:,0] * wh[:,:,1]
    
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area1 = area1.unsqueeze(1).expand(N,M)
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    area2 = area2.unsqueeze(0).expand(N,M)
    
    iou_res = torch.Tensor.float(inter / (area1 + area2 - inter))

    return iou_res

box1 = torch.randint(100,(6,4))
box2 = torch.randint(100,(5,4))
res = iou(box1,box2)

class FPN(nn.Module):
    '''Inchannels = [256,512,1024,2048] , Outchannels = 256'''
    def __init__(self,In_channels , Out_channels) -> None:
        super().__init__()
        
        self.laterals = nn.ModuleList()
        self.fpnconvs = nn.ModuleList()
        
        for i in range(len(In_channels)):
            lateral = nn.Conv2d(in_channels= In_channels[i], out_channels=Out_channels,kernel_size=1)
            fpnconv = nn.Conv2d(in_channels=Out_channels, out_channels=Out_channels,kernel_size=3,padding=1)
            self.laterals.append(lateral)
            self.fpnconvs.append(fpnconv)
    
    def forward(self , inputs):
        '''横向连接统一通道数'''
        laterals_res = [lateral(inputs[i]) for i,lateral in enumerate(self.laterals)]
        '''自顶向下进行加权'''
        used_stages = len(inputs)
        for i in range(used_stages-1,0,-1):
            prev = laterals_res[i-1].shape[2:]
            laterals_res[i-1] += nn.functional.interpolate(laterals_res[i],size=prev,mode = 'nearest')
        '''再通过fpn的3*3卷积'''
        outs = [self.fpnconvs[i](laterals_res[i]) for i in range(used_stages)]
        '''检测任务的话,再下采样到P6'''
        outs.append(nn.functional.max_pool2d(outs[-1],1,stride = 2))
        return outs 


def NMS(box , scores,threshold = 0.5):
    print('hello')
    
        
        
box1 = torch.randint(100,(6,4))
score = torch.rand(6)
NMS(box1,score)