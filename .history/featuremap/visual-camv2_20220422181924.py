import torch
torch.__version__
import re
import cv2
import numpy as np
# import torch
from skimage import io
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import os
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            )
from mmdet.models import build_detector

cfg = Config.fromfile("/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/v1005_roitrans_r50_without_bn.py")
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

checkpoint = load_checkpoint(model, "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/epoch_69.pth", map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES']

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

# cfg.nms_pre = 1000
# cfg.min_bbox_size = 1
# cfg.nms_thr = 10
# cfg.nms_post = 1000
test_cfg = model.rpn_head.test_cfg

# img_metas = data['img_metas']
####################################################################################3
# 此處重寫了rpn_head.get_bboxes，因爲原函數使用detach把gradients的backprop刪了無法backward（）
#######################################################################################

def get_bboxes_tmp(rpn_outs, img_metas, test_cfg):
    
    with_nms = False
    rescale = False
    cls_scores = rpn_outs[0]
    bbox_preds = rpn_outs[1]

    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = model.rpn_head.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [
            cls_scores[i][img_id] for i in range(num_levels) # 此處不再使用detach()
        ]
        bbox_pred_list = [
            bbox_preds[i][img_id] for i in range(num_levels) # 此處不再使用detach()
        ]
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']

        if with_nms:
            # some heads don't support with_nms argument
            proposals = model.rpn_head._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, test_cfg, rescale)
        else:
            proposals = model.rpn_head._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, test_cfg, rescale)
        result_list.append(proposals)
    
    return result_list

def get_bboxes_tmp2(rpn_outs,
                img_metas,
                cfg=None,
                rescale=False):
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
            cls_scores[i][img_id] for i in range(num_levels) #梯度保留 没有用detach
        ]
        bbox_pred_list = [
            bbox_preds[i][img_id] for i in range(num_levels)
        ]
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        proposals = model.rpn_head._get_bboxes_single(cls_score_list, bbox_pred_list,
                                            mlvl_anchors, img_shape,
                                            scale_factor, cfg, rescale)
        # proposals = proposals.to(device='cuda')
        result_list.append(proposals)
    return result_list

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]
        # print('gradient:', self.gradient)

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
#         module = self.net.rpn.head.bbox_pred
#         self.handlers.append(module.register_forward_hook(self._get_features_hook))
#         self.handlers.append(module.register_backward_hook(self._get_grads_hook))
            
                # print(self.handlers)

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        # output = self.net([inputs])
        x = self.net.extract_feat(inputs['img'][0])
        # x = [item.to('cuda') for item in x]
        rpn_outs = self.net.rpn_head(x)
        result_list = get_bboxes_tmp2(rpn_outs,inputs['img_metas'],self.net.rpn_head.test_cfg)
        res= self.net.roi_head.simple_test_for_cam(
            x, result_list,inputs['img_metas'], self.net.roi_head.test_cfg)
        # print(output)
        # score = res[0][0][index][4]
        score = res[0][index][5]
        # proposal_idx = output[0]['labels'][index]  # box来自第几个proposal
        # print(score)
        score.backward()
        # print('gradient:', self.gradient)

        # gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        gradient = self.gradient.cpu().data.numpy().squeeze()
        
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        # feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]
        feature = self.feature.cpu().data.numpy().squeeze()

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # print(cam.shape)
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        # box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        # box = res[0][0][index][:-1].detach().numpy().astype(np.int32)
        box = res[0][index][:-1].detach().numpy().astype(np.float32)
          
        # cam = cv2.resize(cam, (x2 - x1, y2 - y1))
        # cam = cv2.resize(cam, (y2 - y1, x2 - x1)).T
        # print(cam.shape)

        # class_id = output[0]['instances'].pred_classes[index].detach().numpy()
        class_id = res[1][index].detach().numpy()
        plt.imshow(cam)
        return cam, box, class_id

grad_cam = GradCAM(model, 'backbone.layer4')

mask, box, class_id = grad_cam(data[0],0)

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap

image_dict = {}
im = data[0]['img'][0].squeeze(0).permute(1,2,0)
mask = cv2.resize(mask, (im.shape[1], im.shape[0]))
image_cam, image_dict['heatmap'] = gen_cam(im, mask)
cv2.imwrite('/home/guobo/OBBDetection/featuremap/test999.png',image_cam)