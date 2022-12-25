'''
if you use two-stage detector, such as faster rcnn,please change the codes :
1. mmdet/models/detectors/two_stage.py

    def extract_feat(self, img):
    """Directly extract features from the backbone+neck
    """
    x_backbone = self.backbone(img)
    if self.with_neck:
        x_fpn = self.neck(x_backbone)
    return x_backbone,x_fpn

and:

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x_backbone,x_fpn = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x_fpn, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x_fpn, proposal_list, img_metas, rescale=rescale),x_backbone,x_fpn

2.mmdet/apis/inference.py

    def inference_detector(model, img):
    .......
            # forward the model
        with torch.no_grad():
            result,x_backbone,x_fpn= model(return_loss=False, rescale=True, **data)
        return result,x_backbone,x_fpn

if you use other detectors, it is easy to achieve it like this

'''

from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os


def main():

    config = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/v1005_roitrans_r50_without_bn.py"
    checkpoint = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/epoch_69.pth"
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    img = '/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/train/images/000081.png'
    image = cv2.imread(img)
    height, width, channels = image.shape
    result, x_fpn = inference_detector(model, img)

    if not os.path.exists('/home/guobo/OBBDetection/feature_map'):
        os.makedirs('/home/guobo/OBBDetection/feature_map')

    P = x_fpn[0]
    P = P.cpu().detach().numpy()
    P = P.squeeze(0)
    P = np.sum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    print(P.shape)
    cam_2 = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_2), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image_2 = np.uint8(255 * heatmap)
    result_2 = cv2.addWeighted(image, 0.8, heatmap_image_2, 0.3, 0)

    P = x_fpn[1]
    P = P.cpu().detach().numpy()
    P = P.squeeze(0)
    P = np.sum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    print(P.shape)
    cam_3 = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_3), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image_3 = np.uint8(255 * heatmap)
    result_3 = cv2.addWeighted(image, 0.8, heatmap_image_3, 0.3, 0)

    P = x_fpn[2]
    P = P.cpu().detach().numpy()
    P = P.squeeze(0)
    P = np.sum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    print(P.shape)
    cam_4 = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_4), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image_4 = np.uint8(255 * heatmap)
    result_4 = cv2.addWeighted(image, 0.8, heatmap_image_4, 0.3, 0)

    P = x_fpn[3]
    P = P.cpu().detach().numpy()
    P = P.squeeze(0)
    P = np.sum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    print(P.shape)
    cam_5 = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_5), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image_5 = np.uint8(255 * heatmap)
    result_5 = cv2.addWeighted(image, 0.8, heatmap_image_5, 0.3, 0)

    cam_all = cam_5 + cam_4 + cam_3 + cam_2
    cam_all = (cam_all - np.min(cam_all)) / (np.max(cam_all) - np.min(cam_all))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_all), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image_all = np.uint8(255 * heatmap)
    result_all = cv2.addWeighted(image, 0.8, heatmap_image_all, 0.3, 0)

    heat_maps = np.hstack((heatmap_image_2, heatmap_image_3, heatmap_image_4,
                           heatmap_image_5, heatmap_image_all))
    results = np.hstack((result_2, result_3, result_4, result_5, result_all))

    cv2.imwrite(
        '/home/guobo/OBBDetection/feature_map/' + 'all' + '_heatmap.jpg',
        heat_maps)
    cv2.imwrite(
        '/home/guobo/OBBDetection/feature_map/' + 'all' + '_result.jpg',
        results)


if __name__ == '__main__':
    main()