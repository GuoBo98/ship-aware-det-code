# from mmcv.ops import RoIAlignRotated, roi_align_rotated
from ..mmdet.ops import roi_align_rotated
import torch
import numpy as np
import cv2

rois = []
for i in range(360):
    rois.append([0, 300, 300, 100, 300, np.pi * i / 180])
img = cv2.imread("/home/guobo/OBBDetection/rroi_align_test/1.jpg")
# 图像在内存里的形状必须经过contiguous，否则输出会混乱
img = torch.tensor(img, dtype=torch.double, device="cuda").unsqueeze(0).permute(0, 3, 1, 2).contiguous()
rois = torch.tensor(rois, dtype=torch.double, device="cuda")
rois_output = roi_align_rotated(img, rois, (300, 100), 1, 1, True)
for idx, roi in enumerate(rois_output):
    roi_ = roi.permute(1,2,0).to("cpu").numpy().astype(np.uint8)
    cv2.imwrite("/home/guobo/OBBDetection/rroi_align_test/roi_{}.png".format(idx), roi_)