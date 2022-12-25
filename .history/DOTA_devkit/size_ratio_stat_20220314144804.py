import os
import cv2

img_path = '/data2/guobo/01_SHIPRSDET/new-COCO-Format/train_crop_refine/'
img_list = os.listdir(img_path)

for img in img_list:
    image = cv2.imread(os.path.join(img_path,img))
    