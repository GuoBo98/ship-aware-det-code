from mmdet.models.builder import DETECTORS
from .obb_two_stage import OBBTwoStageDetector


@DETECTORS.register_module()
class RoITransformer_PCB(OBBTwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RoITransformer_PCB, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
