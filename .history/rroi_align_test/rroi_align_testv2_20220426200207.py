from mmdet.ops import RoIAlignRotated, roi_align_rotated
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET

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

    idx,cx, cy, w, h, theta = thetaobb

    rect = ((cx, cy), (w, h), theta / np.pi * 180.0)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    cv2.drawContours(img, [rect], -1, color, 2)

    return img


if __name__ == '__main__':
    
    img_metas = np.load("/home/guobo/OBBDetection/split_roi/img_metas.npy",allow_pickle=True)
    img_metas = img_metas.tolist()
    bbox_rois = torch.load("/home/guobo/OBBDetection/split_roi/bbox_roi.pt") 
    obb_rois = torch.load("/home/guobo/OBBDetection/split_roi/obb_roi.pt")
    img = cv2.imread(img_metas[0]['filename'])
    obb_rois = obb_rois[0:3]
    
    for i in range(len(obb_rois)):
        obb_roi = obb_rois[i]
        
        bbox_numpy = obb_roi.cpu().numpy()
        bbox_list = bbox_numpy.tolist()
        image0_obb = show_thetaobb(img,bbox_list,COLORS['Blue'])        
    cv2.imwrite('/home/guobo/OBBDetection/rroi_align_test/show-image0_obb.png',image0_obb)
    
    # 图像在内存里的形状必须经过contiguous，否则输出会混乱
    img = torch.tensor(img, dtype=torch.double, device="cuda").unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    rois = torch.tensor(obb_rois, dtype=torch.double, device="cuda")
    rois_output = roi_align_rotated(img, rois, (60, 500), 1, 1, True)
    for idx, roi in enumerate(rois_output):
        roi_ = roi.permute(1,2,0).to("cpu").numpy().astype(np.uint8)
        cv2.imwrite("/home/guobo/OBBDetection/rroi_align_test/roi_{}.png".format(idx), roi_)