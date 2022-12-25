from mmdet.ops import RoIAlignRotated, roi_align_rotated
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def parse_obj(xml_path):

    tree = ET.parse(xml_path)
    objects = []
    theta_objects = []

    for obj in tree.findall('object'):

        name = obj.find('name').text
        robndBox = obj.find('robndbox')
        cx = float(robndBox.find('cx').text)
        cy = float(robndBox.find('cy').text)
        w = float(robndBox.find('w').text)
        h = float(robndBox.find('h').text)
        angle = float(robndBox.find('angle').text)
        thetaobb = [0,cx, cy, w, h, angle]
        theta_objects.append(thetaobb)

    return theta_objects

if __name__ == '__main__':
    img_path = '/home/guobo/OBBDetection/rroi_align_test/000100.bmp'
    xml_path = '/home/guobo/OBBDetection/rroi_align_test/000100.xml'
    img = cv2.imread(img_path)
    theta_objects = parse_obj(xml_path)
    rois = []
    for obj in theta_objects:
        if obj[3] < obj[4]:
            idx,cx,cy,w,h,angle = obj
            temp = [idx,cx,cy,h,w,angle-np.pi/2]
            rois.append(temp)
    # 图像在内存里的形状必须经过contiguous，否则输出会混乱
    img = torch.tensor(img, dtype=torch.double, device="cuda").unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    rois = torch.tensor(rois, dtype=torch.double, device="cuda")
    rois_output = roi_align_rotated(img, rois, (21, 6), 1, 1, True)
    for idx, roi in enumerate(rois_output):
        roi_ = roi.permute(1,2,0).to("cpu").numpy().astype(np.uint8)
        cv2.imwrite("/home/guobo/OBBDetection/rroi_align_test/roi_{}.png".format(idx), roi_)