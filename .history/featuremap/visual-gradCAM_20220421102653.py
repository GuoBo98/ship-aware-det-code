from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import os

from mmdet.apis import inference_detector, init_detector

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
        rpn_outs = self.net.rpn_head(x)
        result_list = get_bboxes_tmp(rpn_outs,inputs['img_metas'],self.net.rpn_head.test_cfg)
        res= self.net.roi_head.simple_test_bboxes(
            x, inputs['img_metas'], result_list, self.net.roi_head.test_cfg, rescale=True)
        # print(output)
        score = res[0][0][index][4]
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
        box = res[0][0][index][:-1].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        
        # cam = cv2.resize(cam, (x2 - x1, y2 - y1))
        # cam = cv2.resize(cam, (y2 - y1, x2 - x1)).T
        # print(cam.shape)

        # class_id = output[0]['instances'].pred_classes[index].detach().numpy()
        class_id = res[1][0][index].detach().numpy()
        plt.imshow(cam)
        return cam, box, class_id
    
if __name__ == '__main__':
    config = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1307_roitrans_r50_obbGRoI_noBN_noPre/v1307_roitrans_r50_obbGRoI_noBN_noPre.py"
    checkpoint = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1307_roitrans_r50_obbGRoI_noBN_noPre/epoch_69.pth"
    device = 'cuda:0'
    model = init_detector(config, checkpoint, device=device)
    '''选择目标层'''
    target_layer = model.backbone.layer4[-1]
    
    


