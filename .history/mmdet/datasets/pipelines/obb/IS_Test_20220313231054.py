from tkinter.tix import TList
from mmdet.datasets import PIPELINES
import json
import os
import cv2
import tqdm
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors
from copy import copy
from tqdm import trange

def get_heatmap(mask_weight):
    mask_weight_map = mask_weight * 255
    mask_weight_map = mask_weight_map.astype(np.uint8)
    mask_weight_map = cv2.applyColorMap(mask_weight_map, cv2.COLORMAP_HOT)
    return mask_weight_map

Test_path = '/data2/guobo/01_SHIPRSDET/dota-format/IS_TEST/test_height/'
Fuse_path = '/data2/guobo/01_SHIPRSDET/dota-format/IS_TEST/fuse/'
Test_list = os.listdir(Test_path)
img_list, origin_list , switch_list = [],[],[]

for i in trange(len(Test_list)):
    name = Test_list[i].split('-')
    if name[1][0] is 'i':
        img_list.append(Test_list[i])
    if name[1][0] is 'o':
        origin_list.append(Test_list[i])
    if name[1][0] is 's':
        switch_list.append(Test_list[i])

for i in trange(len(img_list)):
    img = cv2.imread(os.path.join(Test_path,img_list[i]))
    name = img_list[i].split('-')[0]
    switch_target = cv2.imread(os.path.join(Test_path,str(name) + '-switch_target.png'))
    origin_bridge = cv2.imread(os.path.join(Test_path,str(name) + '-origin_bridge.png'))
    
    
    switch_h = switch_target.shape[0]
    switch_w = switch_target.shape[1]
    
    #高斯部分
    arr = [4,9,3,9,1.5]
    gaussian_kernel = cv2.getGaussianKernel(switch_h,
                                                    switch_h / arr[0]).reshape(
                                                        (switch_h, ))
    gaussian_kernel /= gaussian_kernel.max()
           
    mask_weight = gaussian_kernel * np.ones((switch_w, switch_h))
    mask_weight = mask_weight.T   
    mask_weight = mask_weight[:, :, None]
    mask_weight = np.concatenate(
        [mask_weight, mask_weight, mask_weight], axis=-1)
    #做截断和归一化    
    th = arr[1]/10 
    mask_weight[mask_weight > th] = th
    mask_weight = mask_weight / th
    mask_weight_map = get_heatmap(mask_weight)
    
    
    gaussian_kernel_w = cv2.getGaussianKernel(switch_w,
                                                    switch_w / arr[2]).reshape(
                                                        (switch_w, ))
    gaussian_kernel_w = gaussian_kernel_w / gaussian_kernel_w.max()
    mask_weight_w = gaussian_kernel_w * np.ones((switch_h, switch_w))
    th2 = arr[3] / 10
    mask_weight_w[mask_weight_w > th2] = th2
    mask_weight_w = mask_weight_w / th2   
    mask_weight_w_map = get_heatmap(mask_weight_w)
    # cv2.imwrite(os.path.join(Fuse_path,str(name) + '-mask_weight_w.png'), mask_weight_w_map)
    
    mask_weight_w = mask_weight_w[:, :, None]
    mask_weight_w = np.concatenate(
        [mask_weight_w, mask_weight_w, mask_weight_w], axis=-1)
    
    # mask_weight_all = mask_weight * mask_weight_w + mask_weight_w
    
    mask_weight_all = mask_weight * mask_weight_w 
    th3 = arr[4] / 10
    mask_weight_all[mask_weight_all>th3] = th3*2
    mask_weight_all = mask_weight_all / (th3*2)
    
    
    
    mask_weight_all_map = get_heatmap(mask_weight_all)
    # cv2.imwrite(os.path.join(Fuse_path,str(name) + '-mask_weight_all.png'), mask_weight_all_map)
    
     

    # cv2.imwrite(os.path.join(Fuse_path,str(name) + '-mask_weight.png'), mask_weight_map)
    
    mask_weight_all_inv = 1 - mask_weight_all

    switch_target_fuse = switch_target * mask_weight_all + origin_bridge * mask_weight_all_inv
    
    htitch= np.hstack((mask_weight_map,mask_weight_w_map,mask_weight_all_map,switch_target, origin_bridge,switch_target_fuse)) # 横向排列\
        
    cv2.imwrite(os.path.join(Fuse_path,str(name) + '-fuse-' + str(arr)+ '.png'), htitch)
    
    
    
    
    
    
    
    
         
    
    