import os
from tkinter import W
import cv2
import matplotlib.pyplot as plt

img_path = '/data2/guobo/01_SHIPRSDET/new-COCO-Format/train_crop_refine/'
img_list = os.listdir(img_path)

size_list= []
ratio_list = []

for img in img_list:
    image = cv2.imread(os.path.join(img_path,img))
    size = image.size
    h = max(image.shape[0:2])
    w = min(image.shape[0:2])
    ratio = h / w 
    size_list.append(size)
    ratio_list.append(ratio)


# size_list.sort()
# plt.bar(range(len(size_list)), size_list)
# plt.savefig('/data2/guobo/01_SHIPRSDET/new-COCO-Format/train_crop_size.png')

ratio_list.sort()
plt.bar(range(len(ratio_list)), ratio_list)
plt.savefig('/data2/guobo/01_SHIPRSDET/new-COCO-Format/train_ratio_list.png')

    