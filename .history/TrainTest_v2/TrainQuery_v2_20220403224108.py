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
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1101_faster_rcnn_orpn_r50_fpn.py 4"
#           )

# '''v1102 使用了IS0.5 ,三张卡 batchsize 3* 2 = 6 lr 0.0075其他和v1101一样'''
# os.system("CUDA_VISIBLE_DEVICES=1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1102_faster_rcnn_orpn_r50_fpn_IS5.py 3"
#           )

'''v1201 三张卡 原始faster-obb'''
# os.system("CUDA_VISIBLE_DEVICES=1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1201_faster_rcnn_obb_r50_fpn.py 3"
#           )

'''v1202 四张卡 faster-obb-IS.5'''
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1202_faster_rcnn_obb_r50_fpn_IS5.py 4"
#           )

'''v1004 加入了IS-0.5 batchsize = 2 * 4 = 8 ,lr=0.01,但是没有BN'''
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1004_roitrans_r50_IS_without_bn.py 4"
#           )


'''v1005-without-bn'''
os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
            /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1005_roitrans_r50_without_bn.py 2"
          )

'''v1006用了双分支boxhead,双卡学习率0.005'''
# os.system("CUDA_VISIBLE_DEVICES=2,3 bash /home/guobo/OBBDetection/tools/dist_train.sh \
#             /home/guobo/OBBDetection/configs/_obb/shipdet_v2/v1006_roitrans_r50_twoBranch.py 2"
#           )
