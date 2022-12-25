from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

annType = ['segm', 'bbox', 'keypoints']

annFile = '/data2/guobo/01_SHIPRSDET/new-COCO-Format/annotations/shipRS_valset.json'
annFile_v2 = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Coco/annotations/shipRS_valset.json"
annFile_height_wdith = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Coco/annotations/shipRS_valset_maskv2.json"

resFile_cascadeMask = '/home/guobo/new-mmdetection/work_dirs/v002_cascade_mask_rcnn_r50_fpn_100e_L3_baseline_1/result_84.segm.json'

v001_roitrans_r50_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v001_faster_rcnn_roitrans_r50_fpn/results-36-mask.json"
v002_100_epoch_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v002_faster_rcnn_roitrans_r50_fpn-100-epoch/results-100-nms-mask.json"
v003_no_pretrain_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v003_faster_rcnn_roitrans_r50_fpn_nopretrain/results-36-nms-mask.json"
v003_cross_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v003_faster_rcnn_roitrans_r50_fpn_nopretrain/results-36-nms-cross-mask.json"
v101_orpn_r50_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v101_faster_rcnn_orpn_r50_fpn/results-36-nms-mask.json"
v102_100_epoch_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v102_faster_rcnn_orpn_r50_fpn-100-epoch/results-100-nms-mask.json"
v401_retinaNet_resFile = '/home/guobo/OBBDetection/TrainTest/work_dirs/v401_retinanet_obb_r50_fpn/results-36-mask.json'
v201_gliding_vertex_r50_resFile = '/home/guobo/OBBDetection/TrainTest/work_dirs/v201_gliding_vertex_r50_fpn/results-36-mask.json'
v301_faster_rcnn_obb_r50_fpn_resFile = '/home/guobo/OBBDetection/TrainTest/work_dirs/v301_faster_rcnn_obb_r50_fpn/results-36-mask.json'
v501_s2anet_r50_fpn_resFile = '/home/guobo/OBBDetection/TrainTest/work_dirs/v501_s2anet_r50_fpn/results-36-mask.json'
v601_fcos_obb_r50_resFile = '/home/guobo/OBBDetection/TrainTest/work_dirs/v601_fcos_obb_r50_caffe_fpn_gn-head_4x4/results-36-mask.json'
v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/results-36-mask.json"
v901_cat_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v901_faster_rcnn_roitrans_r50_fpn_PCB_cutFC_262/results-36-nms-mask.json"
v901_cat_cross_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v901_faster_rcnn_roitrans_r50_fpn_PCB_cutFC_262/results-36-nms-cross-mask.json"
v008_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v008_faster_rcnn_roitrans_r50_fpn_nopretrain/results-68-nms-mask.json"
v007_2_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v007-2_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5/results-75-nms-mask.json"
v009_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v009_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5_gassian/results-72-nms-mask.json"
v010_resFile = "/home/guobo/OBBDetection/TrainTest/work_dirs/v010_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5_gassian_arr/results-68-nms-mask.json"

v1001_resFile = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1001_faster_rcnn_roitrans_r50_fpn/results-99-nms-mask.json"
v1002_resFIle = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1002_roitrans_r50_IS/results-80-nms-mask.json"

v1101_resFile = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1101_faster_rcnn_orpn_r50_fpn/results-95-nms-mask.json"
v1102_resFile = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1102_faster_rcnn_orpn_r50_fpn_IS5/results-75-nms-mask.json"
v1002_resFile_cross = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1002_roitrans_r50_IS/results-80-nms-cross-mask.json"

v1201_resFile = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1201_faster_rcnn_obb_r50_fpn/results-69-nms-mask.json"
v1202_resFile = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1202_faster_rcnn_obb_r50_fpn_IS5/results-93-nms-mask.json"
v1102_resFile_cross = "/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1102_faster_rcnn_orpn_r50_fpn_IS5/results-75-nms-cross-mask.json"

v1005_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/results-68-nms-mask.json"
v1007_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1007_roitrans_r50_IS_without_bn/results-77-nms-mask.json"
v1008_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1008_roitrans_r50_twoBranch_234/results-74-nms-mask.json"

v1010_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1010_roitrans_r50_without_bn_RoIMap112/results-87-nms-mask.json"
v1011_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1011_roitrans_r50_without_bn_RoIMap224/results-80-nms-cross-mask.json"
v1012_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1012_roitrans_r50_without_bn_RoIMap28/results-76-nms-mask.json"

v1301_resFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1301_roitrans_r50_obbGRoI/results-85-nms-mask.json"
v1302_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1302_roitrans_r50_obbGRoI_noBN/results-77-nms-mask.json"
v1303_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1303_roitrans_r50_obbGRoI_noBN_noPrePost/results-69-nms-mask.json"
v1304_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1304_roitrans_r50_obbGRoI_noBN_noPost/results-72-nms-mask.json"
v1305_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1305_roitrans_r50_obbGRoI_noBN_noPrePost_234/results-71-nms-mask.json"
v1306_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1306_roitrans_r50_obbGRoI_noBN_noPrePost_23/results-79-nms-mask.json"
v1307_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1307_roitrans_r50_obbGRoI_noBN_noPre/results-69-nms-mask.json"

v1401_ResFile = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1401_faster_rcnn_roitrans_RRoIAlign/results-80-nms-mask.json"
v1005_resFile_66 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/results-66-nms-mask.json"

v1005_resFile_69 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1005_roitrans_r50_without_bn/results-69-nms-mask.json"
v1501_resFile_76 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1501_ShipIS_GRoI_RRoIAlign_75/results-76-nms-mask.json"

v1101_resFile_95 = "/data2/guobo2021/TrainTest_before0413/TrainTest_v2/work_dirs/v1101_faster_rcnn_orpn_r50_fpn/results-95-nms-mask.json"
v1101_resFile_92 = "/data2/guobo2021/TrainTest_before0413/TrainTest_v2/work_dirs/v1101_faster_rcnn_orpn_r50_fpn/results-92-nms-mask.json"
v1102_resFile_75 = "/data2/guobo2021/TrainTest_before0413/TrainTest_v2/work_dirs/v1102_faster_rcnn_orpn_r50_fpn_IS5/results-75-nms-mask.json"

v1007_roitrans_r50_IS_without_bn_75 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1007_roitrans_r50_IS_without_bn/results-75-nms-mask.json"
v1007_roitrans_r50_IS_without_bn_89 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1007_roitrans_r50_IS_without_bn/results-89-nms-mask.json"

v1601_orpn_75 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1601_faster_rcnn_orpn_r50_fpn/results-75-nms-mask.json"
v1601_orpn_95 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1601_faster_rcnn_orpn_r50_fpn/results-95-nms-mask.json"
v1601_orpn_65 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1601_faster_rcnn_orpn_r50_fpn/results-65-nms-mask.json"
v1601_orpn_64 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1601_faster_rcnn_orpn_r50_fpn/results-64-nms-mask.json"
v1604_orpn_88 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1604_orpn_IS_obbGRoI_RRoIAlign_75/results-88-nms-mask.json"
v1602_orpn_93 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1602_faster_rcnn_orpn_r50_fpn_IS5/results-93-nms-mask.json"

v1503_roiTrans_attention11 = "/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1503_ShipIS_GRoI_RRoIAlign_75_Attention1101/results-83-nms-mask.json"
cocoGT = COCO(annFile_v2)
cocoDT = cocoGT.loadRes(v1503_roiTrans_attention11)

cocoEval = COCOeval(cocoGT, cocoDT, annType[0])

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

categories = [
    'Other-Ship', 'Other-Warship', 'Submarine', 'Other-Aircraft-Carrier',
    'Enterprise', 'Nimitz', 'Midway', 'Ticonderoga', 'Other-Destroyer',
    'Atago-DD', 'Arleigh-Burke-DD', 'Hatsuyuki-DD', 'Hyuga-DD', 'Asagiri-DD',
    'Other-Frigate', 'Perry-FF', 'Patrol', 'Other-Landing', 'YuTing-LL',
    'YuDeng-LL', 'YuDao-LL', 'YuZhao-LL', 'Austin-LL', 'Osumi-LL', 'Wasp-LL',
    'LSD-41-LL', 'LHA-LL', 'Commander', 'Other-Auxiliary-Ship', 'Medical-Ship',
    'Test-Ship', 'Training-Ship', 'AOE', 'Masyuu-AS', 'Sanantonio-AS', 'EPF',
    'Other-Merchant', 'Container-Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat',
    'Ferry', 'Yacht', 'Sailboat', 'Fishing-Vessel', 'Oil-Tanker', 'Hovercraft',
    'Motorboat', 'Dock'
]

voc_map_info_list = []
for i in range(len(categories)):
    stats, _ = cocoEval.summarize_new(catId=i)
    voc_map_info_list.append(stats[0])

for item in voc_map_info_list:
    print(item * 100)

print(np.mean(voc_map_info_list) * 100)
