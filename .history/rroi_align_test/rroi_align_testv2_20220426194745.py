from mmdet.ops import RoIAlignRotated, roi_align_rotated
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET



if __name__ == '__main__':
    
    img_metas = np.load("/home/guobo/OBBDetection/split_roi/img_metas.npy",allow_pickle=True)
    img_metas = img_metas.tolist()
    bbox_rois = torch.load("/home/guobo/OBBDetection/split_roi/bbox_roi.pt") 
    obb_rois = torch.load("/home/guobo/OBBDetection/split_roi/obb_roi.pt")
    # 图像在内存里的形状必须经过contiguous，否则输出会混乱
    img = torch.tensor(img, dtype=torch.double, device="cuda").unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    rois = torch.tensor(rois, dtype=torch.double, device="cuda")
    rois_output = roi_align_rotated(img, rois, (186, 31), 1, 1, True)
    for idx, roi in enumerate(rois_output):
        roi_ = roi.permute(1,2,0).to("cpu").numpy().astype(np.uint8)
        cv2.imwrite("/home/guobo/OBBDetection/rroi_align_test/roi_{}.png".format(idx), roi_)