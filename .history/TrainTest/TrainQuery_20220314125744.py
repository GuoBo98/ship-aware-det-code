import os
'''2021/12/16'''
'''fasterRCNN-roiTrans-r50-pretrained'''
# os.system("PORT=29501 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v001_faster_rcnn_roitrans_r50_fpn.py 2")
'''fasterRCNN-orpn-r50'''
# os.system("PORT=29502 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v101_faster_rcnn_orpn_r50_fpn.py 2")
'''gliding_vertex_r50'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 PORT=29502 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v201_gliding_vertex_r50_fpn.py 2")
'''faster_rcnn_obb_r50'''
# os.system("CUDA_VISIBLE_DEVICES=0,1 PORT=29509 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v301_faster_rcnn_obb_r50_fpn.py 2")
'''retinaNet_obb_r50'''
# os.system("CUDA_VISIBLE_DEVICES=0,1 PORT=29517 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v401_retinanet_obb_r50_fpn.py 2")
'''2021/12/19'''
# '''fasterRCNN-roiTrans-r50-pretrained-100-epoch'''
# os.system("PORT=29506 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v002_faster_rcnn_roitrans_r50_fpn-100-epoch.py 2")

# '''fasterRCNN-orpn-r50'''
# os.system("PORT=29103 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v102_faster_rcnn_orpn_r50_fpn-100-epoch.py 2")
'''2021/12/30'''
'''fasterRCNN-roiTrans-r50-pretrained-PCB_clsscore_weight
   set full + top + down 多加了头和尾的权重，因为我不知道哪个是头哪个是尾，就都加了一遍'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#         /home/guobo/OBBDetection/configs/_obb/shipdet/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight.py 2")
'''fasterRCNN-roiTrans-r50-pretrained-PCB_clsscore_weight
   训100个epoch看看结果如何'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#         /home/guobo/OBBDetection/configs/_obb/shipdet/v702_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight_100epoch.py 2")
# '''fasterRCNN-roiTrans-r50-pretrained-PCB_loss_weight'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#         /home/guobo/OBBDetection/configs/_obb/shipdet/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight.py 2"
#           )
'''fasterRCNN-roiTrans-r50-PCB_loss_weight——newloss
   把之前的四个loss加权成一个loss来回归计算'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#         /home/guobo/OBBDetection/configs/_obb/shipdet/v803_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight_newloss.py 2")

# '''比例改为262，还是采用四个loss回归'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#         /home/guobo/OBBDetection/configs/_obb/shipdet/v804_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight_262.py 2")
'''原始roi-trans,没有预训练'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v003_faster_rcnn_roitrans_r50_fpn_nopretrain.py 2")
'''cutFC_262'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v901_faster_rcnn_roitrans_r50_fpn_PCB_cutFC_262.py 2")
'''v703_no_pretrain'''
# os.system(
#     "CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v703_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight_nopretrain.py 2"
# )
'''v005 fc_bn'''
# os.system(
#     "CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#            /home/guobo/OBBDetection/configs/_obb/shipdet/v005_faster_rcnn_roitrans_r50_fpn_nopretrain_fc_bn.py 2"
# )

# '''加了IS_height,全部的概率'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet/v006_faster_rcnn_roitrans_r50_fpn_IS.py 2")
'''正常裁剪的结果，0.5概率替换，100个epoch'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet/v007_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5.py 4"
#           )

'''想要epoch-74的结果'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet/v007-2_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5.py 4"
#           )

'''完全不替换也没有预训练'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet/v008_faster_rcnn_roitrans_r50_fpn_nopretrain.py 2"
#           )

'''v009用了椭圆高斯'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet/v009_faster_rcnn_roitrans_r50_fpn_IS_crop_0.5_gassian.py 4"
#           )


''''004Test'''
os.system("bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet/v004_faster_rcnn_roitrans_r50_fpn_nopretrain_test.py 2"
          )