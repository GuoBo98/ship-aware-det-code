import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform,mmdet_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import (build_dataloader, build_dataset)
import requests
from PIL import Image
from mmcv import Config, DictAction

config = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/v1005_roitrans_r50_without_bn.py"
checkpoint = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/epoch_69.pth"
device = 'cuda'
cfg = Config.fromfile(config)
model = init_detector(config, checkpoint, device=device)
ship_names = model.CLASSES
'''dataset'''
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

data = []
for i, t in enumerate(data_loader):
    tmp = {}
    tmp['img'] = t['img']
    tmp['img_metas'] = t['img_metas'][0].data[0]
    data.append(tmp)

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(ship_names), 3))

image = data[0]
image_file = cv2.imread(image['img_metas'][0]['filename'])

result = inference_detector(model, image_file)

robboxes = result[1].cpu().numpy()
labels = result[2].cpu().numpy()
classes = [ship_names[item] for item in labels]

target_layers = [model.backbone]
targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=robboxes)]

# define the torchvision image transforms

cam = EigenCAM(
    model,
    target_layers,
    use_cuda=torch.cuda.is_available(),
    reshape_transform=mmdet_reshape_transform)

grayscale_cam = cam(image, targets=targets)
# Take the first image in the batch:
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# And lets draw the boxes again:
image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
Image.fromarray(image_with_bounding_boxes)