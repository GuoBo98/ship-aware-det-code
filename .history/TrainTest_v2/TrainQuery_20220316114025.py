import os
'''2021/12/16'''
'''fasterRCNN-roiTrans-r50-pretrained'''

'''v011-多加入了一些类别，并且限制了被替换的目标的大小和纵横比 arr = [3.5,9,4,9,2.5,2] 学习率0.01'''
os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v10_0_1_faster_rcnn_roitrans_r50_fpn.py.py 2"
          )


