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