import os
from unicodedata import category
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import trange

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def parse_obj(xml_path, filename):

    tree = ET.parse(xml_path + filename + '.xml')
    objects = []

    for obj in tree.findall('object'):

        name = obj.find('name').text
        robndBox = obj.find('robndbox')
        cx = float(robndBox.find('cx').text)
        cy = float(robndBox.find('cy').text)
        w = float(robndBox.find('w').text)
        h = float(robndBox.find('h').text)
        angle = float(robndBox.find('angle').text)
        thetaobb = [cx, cy, w, h, angle]
        pointobb = thetaobb2pointobb(thetaobb)
        bbox = pointobb2bbox(pointobb)
        bbox.append(name)
        objects.append(bbox)

    return objects

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

if __name__ == '__main__':
    
    train_image_path = '/home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/results-36-show/'
    train_xml_path = '/home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/results-36-voc/'
    train_patch_save_path = '/home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/obj-patch/'
    
    if not os.path.exists(train_patch_save_path):
        os.makedirs(train_patch_save_path)
    
    train_image_list = os.listdir(train_image_path)
    train_xml_list = os.listdir(train_xml_path)
    
    for i in trange(len(train_image_list)):
        
        image =  train_image_list[i]
        img_name = image[0:-4]
        img = cv2.imread(os.path.join(train_image_path,image))
        height, width, _ = img.shape
        objs = parse_obj(train_xml_path,img_name)
        for id,obj in enumerate(objs):
            lt_x = int(max(obj[0]-5, 0))
            lt_y = int(max(obj[1]-5, 0))
            rb_x = int(min(obj[2]+5, width))
            rb_y = int(min(obj[3]+5, height))
            category_name = obj[4]
            img_cut = img[lt_y:rb_y, lt_x:rb_x, :]
            save_path = os.path.join(train_patch_save_path,category_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path,img_name+'-'+str(id+1)+'.png'),img_cut)
             
            
        
        
    
    