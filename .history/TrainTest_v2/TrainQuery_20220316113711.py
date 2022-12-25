import os
'''2021/12/16'''
'''fasterRCNN-roiTrans-r50-pretrained'''

'''v011-多加入了一些类别，并且限制了被替换的目标的大小和纵横比 arr = [3.5,9,4,9,2.5,2] 学习率0.01'''
os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet/v011_faster_rcnn_roitrans_r50_fpn_IS_gassian_arr.py 2"
          )


