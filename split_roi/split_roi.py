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
    

if __name__ == '__main__':
    
    img_metas = np.load("/home/guobo/OBBDetection/split_roi/img_metas.npy",allow_pickle=True)
    img_metas = img_metas.tolist()
    bbox_rois = torch.load("/home/guobo/OBBDetection/split_roi/bbox_roi.pt") 
    obb_rois = torch.load("/home/guobo/OBBDetection/split_roi/obb_roi.pt")
    
    # pi = math.pi
    # weight_top,weight_middle,weight_down = [0.3,0.4,0.3]
    # obb_rois_top = obb_rois.new_zeros(obb_rois.shape)
    # obb_rois_middle = obb_rois.new_zeros(obb_rois.shape)
    # obb_rois_down = obb_rois.new_zeros(obb_rois.shape)
    
    # # '''split roi'''
    # # for i in range(len(obb_rois)):
    # #     obb_roi = obb_rois[i]
    # #     roi_idx,x,y,w,h,theta = obb_roi
    # #     tensor_middle = torch.tensor([roi_idx,x,y,w*weight_middle,h,theta])
    # #     if theta < (pi/2) and theta > 0 :
    # #         x_top = x - ((weight_top + weight_middle)*w/2) * abs(math.cos(theta))
    # #         y_top = y - ((weight_top + weight_middle)*w/2) * abs(math.sin(theta))
    # #         w_top = w*weight_top
            
    # #         x_down = x + ((weight_down + weight_middle)*w/2) * abs(math.cos(theta))
    # #         y_down = y + ((weight_down + weight_middle)*w/2) * abs(math.sin(theta))
    # #         w_down = w*weight_down
    # #     else:
    # #         x_top = x + ((weight_top + weight_middle)*w/2) * abs(math.cos(theta))
    # #         y_top = y - ((weight_top + weight_middle)*w/2) * abs(math.sin(theta))
    # #         w_top = w*weight_top
            
    # #         x_down = x - ((weight_down + weight_middle)*w/2) * abs(math.cos(theta))
    # #         y_down = y + ((weight_down + weight_middle)*w/2) * abs(math.sin(theta))
    # #         w_down = w*weight_down
    # #     tensor_top = torch.tensor([roi_idx,x_top,y_top,w_top,h,theta])
    # #     tensor_down = torch.tensor([roi_idx,x_down,y_down,w_down,h,theta])
        
    # #     obb_rois_top[i] = tensor_top
    # #     obb_rois_middle[i] = tensor_middle
    # #     obb_rois_down[i] = tensor_down
    
    obb_rois_top,obb_rois_middle,obb_rois_down = split_obb_rois(obb_rois)
    
    image0_obb_rois = obb_rois[10:12,1:] 
    image0_obb_rois_top = obb_rois_top[10:12,1:] 
    image0_obb_rois_middle = obb_rois_middle[10:12,1:] 
    image0_obb_rois_down = obb_rois_down[10:12,1:] 
    image0_bbox_rois = bbox_rois[10:12,1:]
    
    print('obb-rois is ', image0_obb_rois)
    print('hbb-rois is ', image0_bbox_rois)
    
    image0_bbox = cv2.imread(img_metas[0]['filename'])
    image0_obb = image0_bbox
    
    for i in range(len(image0_obb_rois)):
        obb_roi = image0_obb_rois[i]
        obb_roi_top = image0_obb_rois_top[i]
        obb_roi_down = image0_obb_rois_down[i]
        obb_roi_middle = image0_obb_rois_middle[i]
        
        bbox_numpy = obb_roi_top.cpu().numpy()
        bbox_list = bbox_numpy.tolist()
        image0_obb = show_thetaobb(image0_obb,bbox_list,COLORS['Red'])
        
        bbox_numpy = obb_roi_middle.cpu().numpy()
        bbox_list = bbox_numpy.tolist()
        image0_obb = show_thetaobb(image0_obb,bbox_list,COLORS['Green'])
        
        bbox_numpy = obb_roi_down.cpu().numpy()
        bbox_list = bbox_numpy.tolist()
        image0_obb = show_thetaobb(image0_obb,bbox_list,COLORS['Blue'])
            
    cv2.imwrite('/home/guobo/OBBDetection/split_roi/show-image0_obb.png',image0_obb)
    
    for i in range(len(image0_bbox_rois)):
        hbb_roi = image0_bbox_rois[i]
        bbox_numpy = hbb_roi.cpu().numpy()
        bbox_list = bbox_numpy.tolist()
        image0_bbox = cv2.rectangle(image0_bbox,(int(bbox_list[0]),int(bbox_list[1]),int(bbox_list[2]),int(bbox_list[3])),COLORS['Green'], 1)
    cv2.imwrite('/home/guobo/OBBDetection/split_roi/show-image0_bbox.png',image0_bbox)
    