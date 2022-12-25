import torch
import numpy as np
import cv2
import math

from torch import tensor
from torch._C import device, dtype

COLORS = {
    'Red': (75, 25, 230),
    'Yellow': (25, 225, 225),
    'Green': (75, 180, 60),
    'Blue': (200, 130, 0)
}

def show_thetaobb(img, thetaobb, color):
    """show single theteobb

    Args:
        im (np.array): input image
        thetaobb (list): [cx, cy, w, h, theta]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: image with thetaobb
    """

    cx, cy, w, h, theta = thetaobb

    rect = ((cx, cy), (w, h), theta / np.pi * 180.0)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    cv2.drawContours(img, [rect], -1, color, 2)

    return img

def split_obb_rois(obb_rois):
    
    '''
    This function is to split obb_rois as 3 parts: top,middle,down with the weight of weight_top,weight_middle,weight_down
    
    input : obb_rois
    out_put : 3 splited obb_rois
    
    '''
    pi = torch.tensor(math.pi).cuda()
    weight_top,weight_middle,weight_down = torch.tensor([0.3,0.4,0.3],dtype=torch.float32,device='cuda').cuda()
    obb_rois_top = obb_rois.new_zeros(obb_rois.shape)
    obb_rois_middle = obb_rois.new_zeros(obb_rois.shape)
    obb_rois_down = obb_rois.new_zeros(obb_rois.shape)
    
    '''split roi'''
    for i in range(len(obb_rois)):
        obb_roi = obb_rois[i]
        roi_idx,x,y,w,h,theta = obb_roi
        tensor_middle = torch.tensor([roi_idx,x,y,w*weight_middle,h,theta],dtype=torch.float32,device='cuda')
        '''theta can determin the direction of the obb_roi'''
        if theta < (pi/2) and theta > 0 :
            x_top = x - ((weight_top + weight_middle)*w/2) * abs(math.cos(theta))
            y_top = y - ((weight_top + weight_middle)*w/2) * abs(math.sin(theta))
            w_top = w*weight_top
            
            x_down = x + ((weight_down + weight_middle)*w/2) * abs(math.cos(theta))
            y_down = y + ((weight_down + weight_middle)*w/2) * abs(math.sin(theta))
            w_down = w*weight_down
        else:
            x_top = x + ((weight_top + weight_middle)*w/2) * abs(math.cos(theta))
            y_top = y - ((weight_top + weight_middle)*w/2) * abs(math.sin(theta))
            w_top = w*weight_top
            
            x_down = x - ((weight_down + weight_middle)*w/2) * abs(math.cos(theta))
            y_down = y + ((weight_down + weight_middle)*w/2) * abs(math.sin(theta))
            w_down = w*weight_down
        tensor_top = torch.tensor([roi_idx,x_top,y_top,w_top,h,theta],dtype=torch.float32,device='cuda')
        tensor_down = torch.tensor([roi_idx,x_down,y_down,w_down,h,theta],dtype=torch.float32,device='cuda')
        
        obb_rois_top[i] = tensor_top
        obb_rois_middle[i] = tensor_middle
        obb_rois_down[i] = tensor_down
    
    return obb_rois_top,obb_rois_middle,obb_rois_down
    

def split_roisv2(obb_rois):
   
    '''定义top、middle、down的初始值'''
    weight_top,weight_middle,weight_down = torch.tensor([0.3,0.4,0.3],dtype=torch.float32,device='cuda').cuda()
    
    '''逆时针是小于0,顺时针是大于0'''
    idx_greater =  torch.where(obb_rois[:,5:] > 0) 
    idx_lower = torch.where(obb_rois[:,5:] < 0)  
    greater_obb_rois = obb_rois[idx_greater[0]]
    lower_obb_rois = obb_rois[idx_lower[0]]
    
    '''开始运算顺时针部分,theta>0部分'''
    greater_obb_rois_top , greater_obb_rois_middle , greater_obb_rois_down = greater_obb_rois.clone(),greater_obb_rois.clone(),greater_obb_rois.clone()
    
    greater_obb_rois_top[:,1] = greater_obb_rois_top[:,1] - ((weight_top + weight_middle)*greater_obb_rois_top[:,3]/2) * torch.abs(torch.cos(greater_obb_rois_top[:,5]))
    greater_obb_rois_top[:,2] = greater_obb_rois_top[:,2] - ((weight_top + weight_middle)*greater_obb_rois_top[:,3]/2) * torch.abs(torch.sin(greater_obb_rois_top[:,5]))
    greater_obb_rois_top[:,3] = greater_obb_rois_top[:,3] * weight_top

    greater_obb_rois_down[:,1] = greater_obb_rois_down[:,1] + ((weight_top + weight_middle)*greater_obb_rois_down[:,3]/2) * torch.abs(torch.cos(greater_obb_rois_down[:,5]))
    greater_obb_rois_down[:,2] = greater_obb_rois_down[:,2] + ((weight_top + weight_middle)*greater_obb_rois_down[:,3]/2) * torch.abs(torch.sin(greater_obb_rois_down[:,5]))
    greater_obb_rois_down[:,3] = greater_obb_rois_down[:,3] * weight_down
    
    greater_obb_rois_middle[:,3] = greater_obb_rois_middle[:,3] * weight_middle
    
    '''开始运算逆时针部分,theta<0部分'''
    
    lower_obb_rois_top , lower_obb_rois_middle , lower_obb_rois_down = lower_obb_rois.clone(),lower_obb_rois.clone(),lower_obb_rois.clone()
    
    lower_obb_rois_top[:,1] = lower_obb_rois_top[:,1] + ((weight_top + weight_middle)*lower_obb_rois_top[:,3]/2) * torch.abs(torch.cos(lower_obb_rois_top[:,5]))
    lower_obb_rois_top[:,2] = lower_obb_rois_top[:,2] - ((weight_top + weight_middle)*lower_obb_rois_top[:,3]/2) * torch.abs(torch.sin(lower_obb_rois_top[:,5]))
    lower_obb_rois_top[:,3] = lower_obb_rois_top[:,3] * weight_top

    lower_obb_rois_down[:,1] = lower_obb_rois_down[:,1] - ((weight_top + weight_middle)*lower_obb_rois_down[:,3]/2) * torch.abs(torch.cos(lower_obb_rois_down[:,5]))
    lower_obb_rois_down[:,2] = lower_obb_rois_down[:,2] + ((weight_top + weight_middle)*lower_obb_rois_down[:,3]/2) * torch.abs(torch.sin(lower_obb_rois_down[:,5]))
    lower_obb_rois_down[:,3] = lower_obb_rois_down[:,3] * weight_down
    
    lower_obb_rois_middle[:,3] = lower_obb_rois_middle[:,3] * weight_middle   
    
    '''将两部分给concate起来'''
    
    obb_rois_top = torch.cat([greater_obb_rois_top,lower_obb_rois_top],0)
    obb_rois_middle = torch.cat([greater_obb_rois_middle,lower_obb_rois_middle],0)
    obb_rois_down = torch.cat([greater_obb_rois_down,lower_obb_rois_down],0)
    
    return obb_rois_top,obb_rois_middle,obb_rois_down
    


if __name__ == '__main__':
    
    img_metas = np.load("/home/guobo/OBBDetection/split_roi/img_metas.npy",allow_pickle=True)
    img_metas = img_metas.tolist()
    bbox_rois = torch.load("/home/guobo/OBBDetection/split_roi/bbox_roi.pt") 
    obb_rois = torch.load("/home/guobo/OBBDetection/split_roi/obb_roi.pt")
    
    split_roisv2(obb_rois)
    
     
    
    

    