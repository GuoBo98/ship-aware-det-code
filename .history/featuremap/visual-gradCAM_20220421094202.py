from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import os

from mmdet.apis import inference_detector, init_detector

config = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1307_roitrans_r50_obbGRoI_noBN_noPre/v1307_roitrans_r50_obbGRoI_noBN_noPre.py"
checkpoint = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1307_roitrans_r50_obbGRoI_noBN_noPre/epoch_69.pth"
device = 'cuda:0'
# build the model from a config file and a checkpoint file
model = init_detector(config, checkpoint, device=device)
# test a single image
img = '/home/guobo/OBBDetection/featuremap/000025.bmp'
image = cv2.imread(img)
height, width, channels = image.shape
result, x_fpn = inference_detector(model, img)

