
import torch
from torch import Tensor


box1 = torch.randint(100,size = (5,4))
for item in box1:
    if item[0] > item[2]:
        temp = torch.tensor(item[2])
        item[2] = item[0]
        item[0] = temp
    if item[1] > item[3]:
        temp = torch.tensor(item[3])
        item[3] = item[1]
        item[1] = temp
        
box2 = torch.randint(100,size = (6,4))

for item in box2:
    if item[0] > item[2]:
        temp = torch.tensor(item[2])
        item[2] = item[0]
        item[0] = temp
    if item[1] > item[3]:
        temp = torch.tensor(item[3])
        item[3] = item[1]
        item[1] = temp

def iou(box1:Tensor,box2:Tensor):
    '''假设box1的维度是[N,4],box2的维度是[M,4]'''
    
    N = box1.size(0)
    M = box2.size(0)
    
    lt = torch.max(box1[:,:2].unsqueeze(1).expand(N,M,2) , box2[:,:2].unsqueeze(0).expand(N,M,2))
    br = torch.min(box1[:,2:].unsqueeze(1).expand(N,M,2) , box2[:,2:].unsqueeze(0).expand(N,M,2))
    
    wh = lt - br 
    wh[ wh<0 ] = 0
    
    inter = wh[:,:,0] * wh[:,:,1]
    Area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    Area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    Area1 = Area1.unsqueeze(1).expand(N,M)
    Area2 = Area2.unsqueeze(0).expand(N,M)
    
    iou_res =  torch.Tensor.float(inter / (Area1 + Area2 - inter)) 
    
    return iou_res
    

box1 = torch.randint(100,size = (20,4))
for item in box1:
    if item[0] > item[2]:
        temp = torch.tensor(item[2])
        item[2] = item[0]
        item[0] = temp
    if item[1] > item[3]:
        temp = torch.tensor(item[3])
        item[3] = item[1]
        item[1] = temp

score = torch.rand(size = [20]).squeeze(0)

def nms(bboxes:Tensor,scores:Tensor,threshold = 0.5) :
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1) * (y2 - y1 )
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0 :
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)
    
        '''计算box[i]与其他各个框之间的IoU'''
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
        iou = inter / (areas[i]+areas[order[1:]]-inter)
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0 :
            break
        order = order[idx + 1 ]
    return torch.LongTensor(keep) 
        
        
    
    
            



nms(box1,score,threshold=0.5)