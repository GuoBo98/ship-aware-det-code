import os

'''v1001-99-epoch'''
os.system("bash /home/guobo/OBBDetection/tools/dist_test.sh \
           /home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1001_faster_rcnn_roitrans_r50_fpn/v1001_faster_rcnn_roitrans_r50_fpn.py \
           /home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1001_faster_rcnn_roitrans_r50_fpn/epoch_99.pth \
           4\
           --format-only \
           --options \
           save_dir=/home/guobo/OBBDetection/TrainTest_v2/work_dirs/v1001_faster_rcnn_roitrans_r50_fpn/results-99/"
          )



