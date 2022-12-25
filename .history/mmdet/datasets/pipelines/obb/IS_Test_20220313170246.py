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

Test_path = '/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_test/'
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

for i in range(len(img_list)):
    img = cv2.imread(os.path.join(Test_path,img_list[i]))
    name = img_list[i].split('-')
    switch_target = cv2.imread(os.path.join(str(Test_path),name + '-switch_target.png'))
    origin_bridge = cv2.imread(os.path.join(str(Test_path),name + '-origin_bridge.png'))
    

print('hello')
    
         
    
    