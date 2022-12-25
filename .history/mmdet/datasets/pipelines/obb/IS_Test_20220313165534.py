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

Test_path = '/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_test/'
Test_list = os.listdir(Test_path)
img_list, origin_list , switch_list = [],[],[]

for i in range(len(Test_list)):
    name = Test_list[i].split('-')
    if name[1][0] is 'i':
        img_list.append(Test_list[i])
    if name[1][0] is 'o':
        origin_list.append(Test_list[i])
    if name[1][0] is 's':
        switch_list.append(Test_list[i])



print('hello')
    
         
    
    