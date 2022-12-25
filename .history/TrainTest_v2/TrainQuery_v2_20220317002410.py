import os


'''v1001 修正后的数据集 RoITrans-baseline batchsize = 2 * 3 = 6 ,lr=0.0075'''
os.system("CUDA_VISIBLE_DEVICES=1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1001_faster_rcnn_roitrans_r50_fpn.py 3"
          )

'''v1002 加入了IS-0.5 batchsize = 2 * 4 = 8 ,lr=0.01'''
