import cv2
import os
import shutil
import json
import numpy as np
import tqdm

COLORS = {
    'Red': (75, 25, 230),
    'Yellow': (25, 225, 225),
    'Green': (75, 180, 60),
    'Blue': (200, 130, 0)
}

def get_distence(x, y):
        return (abs(x[0] - y[0]) ** 2 + abs(x[1] - y[1]) ** 2) ** 0.5

def crop_instance(cnt,img):
    rect = cv2.minAreaRect(cnt)
    cx = rect[0][0]
    cy = rect[0][1]
    w = rect[1][0]
    h = rect[1][1]
    theta = rect[2]

    
    '''for ship i dont need the scale_factor'''
    # scale_factor = (max(w,h)/min(w,h))**0.5
    # if w > h:
    #     rect = ((cx, cy), (w, h * scale_factor), theta)
    # else:
    #     rect = ((cx, cy), (w * scale_factor, h), theta)
    # the order of the box points: bottom left, top left, top right,
    
    
    #draw ro-bbox

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img, [box], -1, COLORS['Green'], 2)
    # get width and height of the detected rectangle
    d1 = int(get_distence(box[0], box[1]))
    d2 = int(get_distence(box[1], box[2]))
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    if d1 > d2:
        dst_pts = np.array([[0, d1-1],
                            [0, 0],
                            [d2-1, 0],
                            [d2-1, d1-1]], dtype="float32")
    else:
        dst_pts = np.array([[0, 0],
                            [d1-1, 0],
                            [d1-1, d2-1],
                            [0, d2-1]], 
                            dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    # print(img,M,(width,height))
    
    warped = cv2.warpPerspective(img, M, (min(d1, d2), max(d1, d2)))
    return warped

img_path = "/data2/guobo/01_SHIPRSDET/new-COCO-Format/trainset/images/"
out_path = "/data2/guobo/01_SHIPRSDET/new-COCO-Format/train_crop/"
json_set = ["/data2/guobo/01_SHIPRSDET/new-COCO-Format/annotations/shipRS_trainset.json"]
# ann_file = open("./annotations/plane_train_K2.json")
# ann_json = json.load(ann_file)
# annotations = ann_json["annotations"]
# num_bridges = len(annotations)
# images = ann_json["images"]
sum_bridges = 0
for j in range(len(json_set)):
    ann_file = open(json_set[j])
    ann_json = json.load(ann_file)
    annotations = ann_json["annotations"]
    num_bridges = len(annotations)
    images = ann_json["images"]
    for i in tqdm.trange(num_bridges):
        instance = annotations[i]
        pointobb = instance["pointobb"]
        # pointobb = instance["segmentation"]
        cat_id = instance["category_id"]        
        image_id = instance["image_id"]
        image_name = images[image_id - 1]["file_name"]

        img = cv2.imread(os.path.join(img_path, image_name.split('.')[0] + '.bmp')) ##### pay attention to the suffix of the file_name
        # print(img)
        # cv2.imshow('img',img)

        pointobb = np.array(pointobb,np.int32).reshape((4,1,2))
        bridge = crop_instance(pointobb,img)
        cv2.imwrite(os.path.join(out_path,str(cat_id) + '-' + str(image_name.split('.')[0]) + '-' + str(i) + ".png"), bridge)

    sum_bridges = sum_bridges + num_bridges  # in view of the consistency of the filename

