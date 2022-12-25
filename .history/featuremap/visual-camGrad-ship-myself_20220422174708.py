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
from mmdet.models import build_detector
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import requests
from PIL import Image
from mmcv import Config, DictAction


def get_bboxes_tmp2(rpn_outs, img_metas, cfg=None, rescale=False):
    """Transform network output for a batch into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W)
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W)
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1. The second item is a
            (n,) tensor where each item is the predicted class labelof the
            corresponding box.

    Example:
        >>> import mmcv
        >>> self = AnchorHead(
        >>>     num_classes=9,
        >>>     in_channels=1,
        >>>     anchor_generator=dict(
        >>>         type='AnchorGenerator',
        >>>         scales=[8],
        >>>         ratios=[0.5, 1.0, 2.0],
        >>>         strides=[4,]))
        >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
        >>> cfg = mmcv.Config(dict(
        >>>     score_thr=0.00,
        >>>     nms=dict(type='nms', iou_thr=1.0),
        >>>     max_per_img=10))
        >>> feat = torch.rand(1, 1, 3, 3)
        >>> cls_score, bbox_pred = self.forward_single(feat)
        >>> # note the input lists are over different levels, not images
        >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
        >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
        >>>                               img_metas, cfg)
        >>> det_bboxes, det_labels = result_list[0]
        >>> assert len(result_list) == 1
        >>> assert det_bboxes.shape[1] == 5
        >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
    """
    rescale = False
    cls_scores = rpn_outs[0]
    bbox_preds = rpn_outs[1]

    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = model.rpn_head.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [
            cls_scores[i][img_id] for i in range(num_levels)  #梯度保留 没有用detach
        ]
        bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_levels)]
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        proposals = model.rpn_head._get_bboxes_single(cls_score_list,
                                                      bbox_pred_list,
                                                      mlvl_anchors, img_shape,
                                                      scale_factor, cfg,
                                                      rescale)
        # proposals = proposals.to(device='cuda')
        result_list.append(proposals)
    return result_list

'''build model'''
cfg = Config.fromfile(
    "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/v1005_roitrans_r50_without_bn.py")
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(
    model,
    "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/epoch_69.pth",
    map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES']
ship_names = model.CLASSES
'''build dataset'''
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
'''inference the image'''
model.zero_grad()
# output = model([inputs])
x = model.extract_feat(image['img'][0])
# x = [item.to('cuda') for item in x]
rpn_outs = model.rpn_head(x)
result_list = get_bboxes_tmp2(rpn_outs, image['img_metas'],
                              model.rpn_head.test_cfg)
res = model.roi_head.simple_test_for_cam(x, result_list, image['img_metas'],
                                         model.roi_head.test_cfg)
robboxes = res[0].detach().cpu().numpy()
labels = res[1].detach().cpu().numpy()
classes = [ship_names[item] for item in labels]

target_layers = [model.backbone]
targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=robboxes)]

# define the torchvision image transforms

cam = EigenCAM(
    model,
    target_layers,
    use_cuda=torch.cuda.is_available(),
    reshape_transform=mmdet_reshape_transform)

grayscale_cam = cam(image['img'][0], targets=targets)
# Take the first image in the batch:
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# And lets draw the boxes again:
image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
Image.fromarray(image_with_bounding_boxes)