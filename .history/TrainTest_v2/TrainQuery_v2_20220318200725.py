import os


'''v1001 修正后的数据集 RoITrans-baseline batchsize = 2 * 3 = 6 ,lr=0.0075'''
# os.system("CUDA_VISIBLE_DEVICES=1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1001_faster_rcnn_roitrans_r50_fpn.py 3"
#           )

'''v1002 加入了IS-0.5 batchsize = 2 * 4 = 8 ,lr=0.01'''
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1002_roitrans_r50_IS.py 4"
#           )

'''v1003 加入了IS-0.2 batchsize = 2 * 4 = 8 ,lr=0.01'''
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1003_roitrans_r50_IS_2.py 4"
#           )

# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1003_roitrans_r50_IS_2_continue.py 4"
#           )

'''v1101 修正后的数据集 orpn-baseline batchsize = 2 * 4 = 8 ,lr=0.01'''
os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1003_roitrans_r50_IS_2_continue.py 4"
          )