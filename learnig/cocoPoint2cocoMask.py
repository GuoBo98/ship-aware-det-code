from __future__ import annotations
import os
import json
import cv2
import numpy as np
from tqdm import trange
from pycocotools import mask as coco_mask
import mmcv
from pycocotools.coco import COCO

def pointobb2mask(pointobbs):
        cocoMaskList = []
        for i in trange(len(pointobbs)):
            pointbbx = pointobbs[i]
            if pointbbx['score'] > 0.3:
                newsegm = {}
                newsegm['image_id'] = pointbbx['image_id']  
                newsegm['category_id'] = pointbbx['category_id']
                newsegm['score'] = pointbbx['score']
                '''generate the canvas and write mask on it, then transform the canvas to cocoMask'''       
                canvas = np.zeros((930,930))
                point = pointbbx['pointobbs']
                point = np.reshape(point,(4,2)).astype(int)        
                cv2.fillPoly(canvas,[point],(255,255,255))
                canvas = np.asfortranarray(canvas)    
                cocoMask = coco_mask.encode(canvas.astype(np.uint8))
                '''the type of cocoMask['counts] is byte which is can't be dumped into json'''
                cocoMask['counts'] = str(cocoMask['counts'],'utf-8')
                newsegm['segmentation'] = cocoMask
                cocoMaskList.append(newsegm)
        return cocoMaskList

if __name__ =='__main__':
    
    annFile = '/data2/guobo/01_SHIPRSDET/ShipDetv2/Coco/annotations/shipRS_valset.json'
    
    coco_GT = COCO(annFile)
    
    anns = coco_GT.anns
    
    all_objs = []
    
    for i in range(len(anns)) :
        obj = {}
        ann = anns[i + 1]
        obj['image_id'] = ann['image_id']
        obj['pointobbs'] = ann['pointobb']
        obj['category_id'] = ann['category_id']
        obj['score'] = 1
        all_objs.append(obj)
    
    all_coco_mask = pointobb2mask(all_objs)
    mmcv.dump(all_coco_mask,'/data2/guobo/01_SHIPRSDET/ShipDetv2/Coco/annotations/shipRS_valset_mask.json')
    
        
         