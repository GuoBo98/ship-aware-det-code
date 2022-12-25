_base_ = [
    '../_base_/datasets/shipdataset.py',
    '../_base_/schedules/schedule_3x.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='RoITransformer_PCB',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='RoITransRoIHead_PCB_clsscore_weight',
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='OBBSingleRoIExtractor',
                roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
                out_channels=256,
                extend_factor=(1.4, 1.2),
                featmap_strides=[4, 8, 16, 32]),
        ],
        bbox_head=[
            dict(
                type='OBBShared2FCBBoxHead',
                start_bbox_type='hbb',
                end_bbox_type='obb',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='HBB2OBBDeltaXYWHTCoder',
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='OBBShared2FCBBoxHead',
                start_bbox_type='obb',
                end_bbox_type='obb',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))
        ]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=200,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D')),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='OBBOverlaps')),
            sampler=dict(
                type='OBBRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        score_weights=[0, 1],
        nms=dict(type='obb_nms', iou_thr=0.1), max_per_img=2000))

data = dict(
    samples_per_gpu=4)

# optimizer 2 gpus batchsize=4,lr=0.01
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
load_from = '/home/guobo/OBBDetection/work_dirs/pretrain/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5/epoch_12.pth'

'''
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 791  | 0.5960 | 0.3415 |
| Other-Warship          | 209 | 589  | 0.7464 | 0.4931 |
| Submarine              | 171 | 200  | 0.8421 | 0.7901 |
| Other-Aircraft-Carrier | 4   | 22   | 1.0000 | 0.9091 |
| Enterprise             | 18  | 29   | 0.7778 | 0.7143 |
| Nimitz                 | 19  | 35   | 1.0000 | 0.8881 |
| Midway                 | 5   | 18   | 1.0000 | 0.8333 |
| Ticonderoga            | 77  | 113  | 0.9091 | 0.8780 |
| Other-Destroyer        | 46  | 208  | 0.9348 | 0.5840 |
| Atago-DD               | 41  | 130  | 0.9756 | 0.7878 |
| Arleigh-Burke-DD       | 130 | 227  | 0.9692 | 0.8987 |
| Hatsuyuki-DD           | 22  | 115  | 0.9545 | 0.6824 |
| Hyuga-DD               | 16  | 28   | 1.0000 | 1.0000 |
| Asagiri-DD             | 14  | 104  | 1.0000 | 0.6744 |
| Other-Frigate          | 28  | 121  | 0.8214 | 0.6019 |
| Perry-FF               | 77  | 148  | 0.9610 | 0.8912 |
| Patrol                 | 37  | 82   | 0.7027 | 0.6026 |
| Other-Landing          | 18  | 88   | 0.5556 | 0.3444 |
| YuTing-LL              | 19  | 32   | 0.8947 | 0.7243 |
| YuDeng-LL              | 11  | 26   | 0.8182 | 0.7231 |
| YuDao-LL               | 5   | 19   | 1.0000 | 0.9697 |
| YuZhao-LL              | 6   | 24   | 1.0000 | 1.0000 |
| Austin-LL              | 27  | 61   | 0.9630 | 0.7741 |
| Osumi-LL               | 6   | 31   | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 11   | 1.0000 | 0.9740 |
| LSD-41-LL              | 21  | 74   | 0.9048 | 0.8902 |
| LHA-LL                 | 31  | 68   | 0.9677 | 0.9000 |
| Commander              | 32  | 87   | 1.0000 | 0.9865 |
| Other-Auxiliary-Ship   | 18  | 137  | 0.6111 | 0.3574 |
| Medical-Ship           | 5   | 19   | 0.8000 | 0.7636 |
| Test-Ship              | 7   | 32   | 0.7143 | 0.6465 |
| Training-Ship          | 11  | 33   | 1.0000 | 0.9860 |
| AOE                    | 11  | 52   | 1.0000 | 0.8590 |
| Masyuu-AS              | 8   | 54   | 1.0000 | 1.0000 |
| Sanantonio-AS          | 13  | 45   | 0.9231 | 0.8864 |
| EPF                    | 10  | 29   | 0.9000 | 0.8538 |
| Other-Merchant         | 50  | 156  | 0.4000 | 0.1107 |
| Container-Ship         | 72  | 196  | 0.9028 | 0.6968 |
| RoRo                   | 20  | 58   | 0.9500 | 0.8620 |
| Cargo                  | 169 | 327  | 0.8817 | 0.6755 |
| Barge                  | 22  | 122  | 0.4091 | 0.2323 |
| Tugboat                | 46  | 153  | 0.8043 | 0.5926 |
| Ferry                  | 53  | 203  | 0.7170 | 0.4537 |
| Yacht                  | 140 | 232  | 0.9357 | 0.8023 |
| Sailboat               | 341 | 177  | 0.2581 | 0.2003 |
| Fishing-Vessel         | 99  | 408  | 0.7172 | 0.3479 |
| Oil-Tanker             | 32  | 83   | 0.8438 | 0.7206 |
| Hovercraft             | 31  | 87   | 0.8710 | 0.5703 |
| Motorboat              | 398 | 694  | 0.3794 | 0.1379 |
| Dock                   | 154 | 276  | 0.6039 | 0.5227 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.7027 |
+------------------------+-----+------+--------+--------+
'''