import os
'''2021/12/16'''
'''fasterRCNN-roiTrans-r50-pretrained'''

'''v1001 修正后的数据集'''
os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1001_faster_rcnn_roitrans_r50_fpn.py.py 2"
          )


