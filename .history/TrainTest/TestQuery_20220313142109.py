import os
'''2021/12/16'''
'''fasterRCNN-roiTrans-r50-pretrained'''
# os.system("PORT=12345 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v001_faster_rcnn_roitrans_r50_fpn/v001_faster_rcnn_roitrans_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v001_faster_rcnn_roitrans_r50_fpn/epoch_36.pth \
#            4\
#            --out /home/guobo/OBBDetection/TrainTest/work_dirs/v001_faster_rcnn_roitrans_r50_fpn/results-36.pkl \
#            --eval mAP \
#            --show-dir /home/guobo/OBBDetection/TrainTest/work_dirs/v001_faster_rcnn_roitrans_r50_fpn/results-36-show")
'''fasterRCNN-orpn-r50-pretrained'''
# os.system("PORT=12343 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v101_faster_rcnn_orpn_r50_fpn/v101_faster_rcnn_orpn_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v101_faster_rcnn_orpn_r50_fpn/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v101_faster_rcnn_orpn_r50_fpn/results-36/")
'''RetinaNet'''
# os.system("PORT=12243 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v401_retinanet_obb_r50_fpn/v401_retinanet_obb_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v401_retinanet_obb_r50_fpn/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v401_retinanet_obb_r50_fpn/results-36/")
'''v201_gliding_vertex_r50_fpn'''
# os.system("PORT=12243 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v201_gliding_vertex_r50_fpn/v201_gliding_vertex_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v201_gliding_vertex_r50_fpn/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v201_gliding_vertex_r50_fpn/results-36/")
'''v301_faster_rcnn_obb_r50_fpn'''
# os.system("PORT=12543 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v301_faster_rcnn_obb_r50_fpn/v301_faster_rcnn_obb_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v301_faster_rcnn_obb_r50_fpn/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v301_faster_rcnn_obb_r50_fpn/results-36/")
'''v501_s2anet_r50_fpn'''
# os.system("PORT=12543 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v501_s2anet_r50_fpn/v501_s2anet_r50_fpn.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v501_s2anet_r50_fpn/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v501_s2anet_r50_fpn/results-36/")

# '''v601_FCOS_OBB'''
# os.system("PORT=12543 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v601_fcos_obb_r50_caffe_fpn_gn-head_4x4/v601_fcos_obb_r50_caffe_fpn_gn-head_4x4.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v601_fcos_obb_r50_caffe_fpn_gn-head_4x4/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v601_fcos_obb_r50_caffe_fpn_gn-head_4x4/results-36/")
'''v801 cls+top+middle+down
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 376  | 0.4848 | 0.3057 |
| Other-Warship          | 209 | 310  | 0.6316 | 0.4510 |
| Submarine              | 171 | 145  | 0.7661 | 0.7096 |
| Other-Aircraft-Carrier | 4   | 6    | 1.0000 | 1.0000 |
| Enterprise             | 18  | 15   | 0.7222 | 0.7152 |
| Nimitz                 | 19  | 22   | 0.8947 | 0.7785 |
| Midway                 | 5   | 8    | 1.0000 | 0.8333 |
| Ticonderoga            | 77  | 72   | 0.8961 | 0.8182 |
| Other-Destroyer        | 46  | 109  | 0.8261 | 0.4827 |
| Atago-DD               | 41  | 89   | 0.8780 | 0.7013 |
| Arleigh-Burke-DD       | 130 | 136  | 0.9462 | 0.8888 |
| Hatsuyuki-DD           | 22  | 38   | 0.7273 | 0.5545 |
| Hyuga-DD               | 16  | 19   | 1.0000 | 1.0000 |
| Asagiri-DD             | 14  | 25   | 0.6429 | 0.4782 |
| Other-Frigate          | 28  | 37   | 0.5714 | 0.4973 |
| Perry-FF               | 77  | 92   | 0.9481 | 0.8914 |
| Patrol                 | 37  | 24   | 0.5135 | 0.5145 |
| Other-Landing          | 18  | 21   | 0.3889 | 0.3121 |
| YuTing-LL              | 19  | 22   | 0.8421 | 0.6865 |
| YuDeng-LL              | 11  | 10   | 0.6364 | 0.5758 |
| YuDao-LL               | 5   | 7    | 1.0000 | 0.9481 |
| YuZhao-LL              | 6   | 9    | 1.0000 | 1.0000 |
| Austin-LL              | 27  | 33   | 0.8519 | 0.7569 |
| Osumi-LL               | 6   | 6    | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 7    | 1.0000 | 1.0000 |
| LSD-41-LL              | 21  | 39   | 0.8571 | 0.7693 |
| LHA-LL                 | 31  | 36   | 0.9677 | 0.9032 |
| Commander              | 32  | 48   | 1.0000 | 0.9727 |
| Other-Auxiliary-Ship   | 18  | 32   | 0.4444 | 0.3636 |
| Medical-Ship           | 5   | 8    | 0.8000 | 0.8182 |
| Test-Ship              | 7   | 6    | 0.5714 | 0.4909 |
| Training-Ship          | 11  | 16   | 1.0000 | 0.9924 |
| AOE                    | 11  | 23   | 0.9091 | 0.6353 |
| Masyuu-AS              | 8   | 14   | 1.0000 | 1.0000 |
| Sanantonio-AS          | 13  | 15   | 0.9231 | 0.9091 |
| EPF                    | 10  | 9    | 0.8000 | 0.7879 |
| Other-Merchant         | 50  | 50   | 0.2000 | 0.1296 |
| Container-Ship         | 72  | 90   | 0.6667 | 0.5377 |
| RoRo                   | 20  | 23   | 0.8500 | 0.8045 |
| Cargo                  | 169 | 216  | 0.8225 | 0.7081 |
| Barge                  | 22  | 20   | 0.2273 | 0.2102 |
| Tugboat                | 46  | 69   | 0.6739 | 0.5261 |
| Ferry                  | 53  | 63   | 0.4906 | 0.3957 |
| Yacht                  | 140 | 163  | 0.8714 | 0.7292 |
| Sailboat               | 341 | 24   | 0.0557 | 0.0909 |
| Fishing-Vessel         | 99  | 150  | 0.4646 | 0.2592 |
| Oil-Tanker             | 32  | 34   | 0.5938 | 0.5013 |
| Hovercraft             | 31  | 57   | 0.8387 | 0.5432 |
| Motorboat              | 398 | 407  | 0.2613 | 0.1365 |
| Dock                   | 154 | 73   | 0.4091 | 0.4284 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.6509 |
+------------------------+-----+------+--------+--------+
'''
'''v801 cls
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 817  | 0.6195 | 0.3667 |
| Other-Warship          | 209 | 603  | 0.7560 | 0.4912 |
| Submarine              | 171 | 216  | 0.8304 | 0.7792 |
| Other-Aircraft-Carrier | 4   | 18   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 33   | 0.7778 | 0.7152 |
| Nimitz                 | 19  | 33   | 1.0000 | 0.8984 |
| Midway                 | 5   | 24   | 1.0000 | 0.8333 |
| Ticonderoga            | 77  | 127  | 0.9351 | 0.8949 |
| Other-Destroyer        | 46  | 221  | 0.9565 | 0.3921 |
| Atago-DD               | 41  | 139  | 0.9512 | 0.6878 |
| Arleigh-Burke-DD       | 130 | 201  | 0.9615 | 0.8883 |
| Hatsuyuki-DD           | 22  | 109  | 0.9545 | 0.6669 |
| Hyuga-DD               | 16  | 29   | 1.0000 | 0.9893 |
| Asagiri-DD             | 14  | 106  | 1.0000 | 0.4868 |
| Other-Frigate          | 28  | 133  | 0.6071 | 0.5371 |
| Perry-FF               | 77  | 141  | 0.9481 | 0.9028 |
| Patrol                 | 37  | 112  | 0.8649 | 0.6189 |
| Other-Landing          | 18  | 93   | 0.7222 | 0.3131 |
| YuTing-LL              | 19  | 33   | 0.8947 | 0.6756 |
| YuDeng-LL              | 11  | 29   | 0.7273 | 0.5665 |
| YuDao-LL               | 5   | 13   | 1.0000 | 1.0000 |
| YuZhao-LL              | 6   | 29   | 1.0000 | 1.0000 |
| Austin-LL              | 27  | 57   | 0.9630 | 0.7747 |
| Osumi-LL               | 6   | 27   | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 16   | 1.0000 | 0.9174 |
| LSD-41-LL              | 21  | 76   | 0.9524 | 0.8885 |
| LHA-LL                 | 31  | 59   | 0.9677 | 0.9003 |
| Commander              | 32  | 79   | 1.0000 | 0.9465 |
| Other-Auxiliary-Ship   | 18  | 128  | 0.6111 | 0.3541 |
| Medical-Ship           | 5   | 33   | 0.8000 | 0.8182 |
| Test-Ship              | 7   | 24   | 0.7143 | 0.6515 |
| Training-Ship          | 11  | 35   | 1.0000 | 0.9697 |
| AOE                    | 11  | 60   | 0.9091 | 0.7467 |
| Masyuu-AS              | 8   | 59   | 1.0000 | 0.9126 |
| Sanantonio-AS          | 13  | 41   | 0.9231 | 0.8629 |
| EPF                    | 10  | 19   | 0.8000 | 0.7980 |
| Other-Merchant         | 50  | 177  | 0.3400 | 0.0976 |
| Container-Ship         | 72  | 182  | 0.8333 | 0.6337 |
| RoRo                   | 20  | 42   | 0.9000 | 0.8735 |
| Cargo                  | 169 | 322  | 0.8876 | 0.7045 |
| Barge                  | 22  | 118  | 0.4091 | 0.1847 |
| Tugboat                | 46  | 131  | 0.7826 | 0.5813 |
| Ferry                  | 53  | 195  | 0.6981 | 0.4484 |
| Yacht                  | 140 | 234  | 0.9429 | 0.7933 |
| Sailboat               | 341 | 181  | 0.2522 | 0.2065 |
| Fishing-Vessel         | 99  | 414  | 0.6263 | 0.2708 |
| Oil-Tanker             | 32  | 76   | 0.7812 | 0.6810 |
| Hovercraft             | 31  | 70   | 0.8387 | 0.5654 |
| Motorboat              | 398 | 715  | 0.3668 | 0.1472 |
| Dock                   | 154 | 330  | 0.6558 | 0.5261 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.6792 |
+------------------------+-----+------+--------+--------+
'''
'''cls-top
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 1194 | 0.5859 | 0.2862 |
| Other-Warship          | 209 | 1010 | 0.6555 | 0.3629 |
| Submarine              | 171 | 398  | 0.7485 | 0.6498 |
| Other-Aircraft-Carrier | 4   | 51   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 40   | 0.7222 | 0.6761 |
| Nimitz                 | 19  | 60   | 0.9474 | 0.7904 |
| Midway                 | 5   | 29   | 0.8000 | 0.5134 |
| Ticonderoga            | 77  | 362  | 0.8571 | 0.6916 |
| Other-Destroyer        | 46  | 360  | 0.7391 | 0.2703 |
| Atago-DD               | 41  | 249  | 0.8780 | 0.4532 |
| Arleigh-Burke-DD       | 130 | 435  | 0.9385 | 0.8291 |
| Hatsuyuki-DD           | 22  | 163  | 0.7727 | 0.3369 |
| Hyuga-DD               | 16  | 103  | 1.0000 | 0.9733 |
| Asagiri-DD             | 14  | 126  | 0.8571 | 0.3895 |
| Other-Frigate          | 28  | 263  | 0.6429 | 0.3062 |
| Perry-FF               | 77  | 309  | 0.9481 | 0.8029 |
| Patrol                 | 37  | 125  | 0.5676 | 0.3110 |
| Other-Landing          | 18  | 113  | 0.4444 | 0.2919 |
| YuTing-LL              | 19  | 51   | 0.8947 | 0.6732 |
| YuDeng-LL              | 11  | 44   | 0.5455 | 0.3989 |
| YuDao-LL               | 5   | 16   | 1.0000 | 0.8935 |
| YuZhao-LL              | 6   | 42   | 1.0000 | 0.8310 |
| Austin-LL              | 27  | 138  | 0.8148 | 0.5588 |
| Osumi-LL               | 6   | 77   | 1.0000 | 0.5981 |
| Wasp-LL                | 6   | 48   | 1.0000 | 0.8909 |
| LSD-41-LL              | 21  | 172  | 0.7619 | 0.6157 |
| LHA-LL                 | 31  | 157  | 0.9677 | 0.8767 |
| Commander              | 32  | 178  | 0.9062 | 0.7522 |
| Other-Auxiliary-Ship   | 18  | 130  | 0.6111 | 0.2917 |
| Medical-Ship           | 5   | 47   | 0.8000 | 0.5227 |
| Test-Ship              | 7   | 57   | 0.7143 | 0.4886 |
| Training-Ship          | 11  | 76   | 0.9091 | 0.8843 |
| AOE                    | 11  | 112  | 0.7273 | 0.4275 |
| Masyuu-AS              | 8   | 73   | 0.8750 | 0.6488 |
| Sanantonio-AS          | 13  | 103  | 0.9231 | 0.6854 |
| EPF                    | 10  | 46   | 0.9000 | 0.7841 |
| Other-Merchant         | 50  | 224  | 0.3200 | 0.0726 |
| Container-Ship         | 72  | 343  | 0.6944 | 0.4483 |
| RoRo                   | 20  | 105  | 0.9500 | 0.7149 |
| Cargo                  | 169 | 706  | 0.8107 | 0.5546 |
| Barge                  | 22  | 178  | 0.2727 | 0.1648 |
| Tugboat                | 46  | 164  | 0.7174 | 0.5590 |
| Ferry                  | 53  | 279  | 0.6038 | 0.4056 |
| Yacht                  | 140 | 285  | 0.9000 | 0.7319 |
| Sailboat               | 341 | 213  | 0.1789 | 0.1061 |
| Fishing-Vessel         | 99  | 469  | 0.5657 | 0.2382 |
| Oil-Tanker             | 32  | 149  | 0.5938 | 0.5159 |
| Hovercraft             | 31  | 79   | 0.8065 | 0.5880 |
| Motorboat              | 398 | 847  | 0.3291 | 0.1612 |
| Dock                   | 154 | 890  | 0.3896 | 0.2344 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.5450 |
+------------------------+-----+------+--------+--------+
'''
'''cls-down
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 1146 | 0.5522 | 0.2601 |
| Other-Warship          | 209 | 932  | 0.6938 | 0.4064 |
| Submarine              | 171 | 393  | 0.7661 | 0.6673 |
| Other-Aircraft-Carrier | 4   | 46   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 47   | 0.7778 | 0.6901 |
| Nimitz                 | 19  | 71   | 0.8947 | 0.7216 |
| Midway                 | 5   | 31   | 1.0000 | 0.7556 |
| Ticonderoga            | 77  | 355  | 0.8831 | 0.7287 |
| Other-Destroyer        | 46  | 340  | 0.7609 | 0.2939 |
| Atago-DD               | 41  | 245  | 0.8780 | 0.5868 |
| Arleigh-Burke-DD       | 130 | 403  | 0.9462 | 0.8670 |
| Hatsuyuki-DD           | 22  | 177  | 0.8182 | 0.4384 |
| Hyuga-DD               | 16  | 97   | 1.0000 | 0.9947 |
| Asagiri-DD             | 14  | 131  | 0.8571 | 0.2884 |
| Other-Frigate          | 28  | 252  | 0.7143 | 0.2548 |
| Perry-FF               | 77  | 321  | 0.8831 | 0.7225 |
| Patrol                 | 37  | 150  | 0.7568 | 0.4329 |
| Other-Landing          | 18  | 124  | 0.5000 | 0.3315 |
| YuTing-LL              | 19  | 49   | 0.7895 | 0.4995 |
| YuDeng-LL              | 11  | 41   | 0.4545 | 0.1625 |
| YuDao-LL               | 5   | 21   | 1.0000 | 0.8646 |
| YuZhao-LL              | 6   | 32   | 1.0000 | 1.0000 |
| Austin-LL              | 27  | 117  | 0.8519 | 0.5953 |
| Osumi-LL               | 6   | 73   | 1.0000 | 0.8365 |
| Wasp-LL                | 6   | 37   | 1.0000 | 1.0000 |
| LSD-41-LL              | 21  | 175  | 0.8571 | 0.5449 |
| LHA-LL                 | 31  | 150  | 0.9677 | 0.8484 |
| Commander              | 32  | 162  | 0.9688 | 0.7739 |
| Other-Auxiliary-Ship   | 18  | 158  | 0.5000 | 0.2260 |
| Medical-Ship           | 5   | 60   | 0.8000 | 0.5083 |
| Test-Ship              | 7   | 68   | 0.5714 | 0.3506 |
| Training-Ship          | 11  | 69   | 0.8182 | 0.7043 |
| AOE                    | 11  | 122  | 0.7273 | 0.2798 |
| Masyuu-AS              | 8   | 83   | 1.0000 | 0.8400 |
| Sanantonio-AS          | 13  | 102  | 0.9231 | 0.6571 |
| EPF                    | 10  | 48   | 0.8000 | 0.7980 |
| Other-Merchant         | 50  | 237  | 0.3200 | 0.0895 |
| Container-Ship         | 72  | 331  | 0.7917 | 0.5282 |
| RoRo                   | 20  | 111  | 1.0000 | 0.8839 |
| Cargo                  | 169 | 674  | 0.8343 | 0.5436 |
| Barge                  | 22  | 153  | 0.4545 | 0.2251 |
| Tugboat                | 46  | 166  | 0.7609 | 0.5335 |
| Ferry                  | 53  | 267  | 0.6415 | 0.3662 |
| Yacht                  | 140 | 285  | 0.9214 | 0.7317 |
| Sailboat               | 341 | 176  | 0.1760 | 0.1345 |
| Fishing-Vessel         | 99  | 483  | 0.5960 | 0.2518 |
| Oil-Tanker             | 32  | 141  | 0.6250 | 0.4874 |
| Hovercraft             | 31  | 88   | 0.7742 | 0.6148 |
| Motorboat              | 398 | 816  | 0.3291 | 0.1474 |
| Dock                   | 154 | 858  | 0.3506 | 0.2428 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.5542 |
+------------------------+-----+------+--------+--------+'''
'''cls-middle
Start calculate mAP!!!
Result is Only for reference, final result is subject to DOTA_devkit

+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 820  | 0.6229 | 0.3463 |
| Other-Warship          | 209 | 642  | 0.7512 | 0.4826 |
| Submarine              | 171 | 222  | 0.8304 | 0.7743 |
| Other-Aircraft-Carrier | 4   | 16   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 25   | 0.7778 | 0.7152 |
| Nimitz                 | 19  | 35   | 1.0000 | 0.9125 |
| Midway                 | 5   | 22   | 1.0000 | 0.8333 |
| Ticonderoga            | 77  | 132  | 0.9221 | 0.8922 |
| Other-Destroyer        | 46  | 204  | 0.9130 | 0.4117 |
| Atago-DD               | 41  | 119  | 0.8780 | 0.6574 |
| Arleigh-Burke-DD       | 130 | 196  | 0.9692 | 0.8926 |
| Hatsuyuki-DD           | 22  | 105  | 1.0000 | 0.5233 |
| Hyuga-DD               | 16  | 29   | 1.0000 | 0.9893 |
| Asagiri-DD             | 14  | 96   | 1.0000 | 0.4144 |
| Other-Frigate          | 28  | 146  | 0.6071 | 0.4996 |
| Perry-FF               | 77  | 138  | 0.9481 | 0.8953 |
| Patrol                 | 37  | 106  | 0.7838 | 0.5895 |
| Other-Landing          | 18  | 85   | 0.5000 | 0.3430 |
| YuTing-LL              | 19  | 48   | 0.8947 | 0.6753 |
| YuDeng-LL              | 11  | 24   | 0.6364 | 0.5682 |
| YuDao-LL               | 5   | 8    | 1.0000 | 0.9481 |
| YuZhao-LL              | 6   | 27   | 1.0000 | 1.0000 |
| Austin-LL              | 27  | 52   | 0.9259 | 0.8023 |
| Osumi-LL               | 6   | 17   | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 19   | 0.8333 | 0.8182 |
| LSD-41-LL              | 21  | 99   | 0.9524 | 0.8325 |
| LHA-LL                 | 31  | 62   | 0.9677 | 0.8868 |
| Commander              | 32  | 104  | 1.0000 | 0.9336 |
| Other-Auxiliary-Ship   | 18  | 107  | 0.5556 | 0.3329 |
| Medical-Ship           | 5   | 28   | 0.8000 | 0.8182 |
| Test-Ship              | 7   | 30   | 0.7143 | 0.5523 |
| Training-Ship          | 11  | 40   | 1.0000 | 0.9333 |
| AOE                    | 11  | 69   | 0.9091 | 0.5933 |
| Masyuu-AS              | 8   | 51   | 1.0000 | 1.0000 |
| Sanantonio-AS          | 13  | 55   | 0.9231 | 0.8281 |
| EPF                    | 10  | 18   | 0.8000 | 0.7576 |
| Other-Merchant         | 50  | 162  | 0.3200 | 0.0769 |
| Container-Ship         | 72  | 188  | 0.8333 | 0.5615 |
| RoRo                   | 20  | 55   | 0.9000 | 0.8332 |
| Cargo                  | 169 | 333  | 0.8639 | 0.6673 |
| Barge                  | 22  | 97   | 0.2727 | 0.1539 |
| Tugboat                | 46  | 125  | 0.7609 | 0.5505 |
| Ferry                  | 53  | 200  | 0.6604 | 0.4506 |
| Yacht                  | 140 | 223  | 0.9286 | 0.7621 |
| Sailboat               | 341 | 174  | 0.2757 | 0.2154 |
| Fishing-Vessel         | 99  | 414  | 0.6869 | 0.2855 |
| Oil-Tanker             | 32  | 77   | 0.7500 | 0.6066 |
| Hovercraft             | 31  | 64   | 0.8065 | 0.5802 |
| Motorboat              | 398 | 721  | 0.3643 | 0.1490 |
| Dock                   | 154 | 409  | 0.6234 | 0.4678 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.6563 |
+------------------------+-----+------+--------+--------+
'''

# '''middle + cls'''
# os.system("bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight/epoch_36.pth \
#            2\
#            --eval mAP")
'''统一成一个loss
cls + middle
+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 449  | 0.5185 | 0.3373 |
| Other-Warship          | 209 | 398  | 0.6651 | 0.4400 |
| Submarine              | 171 | 167  | 0.8012 | 0.7849 |
| Other-Aircraft-Carrier | 4   | 11   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 26   | 0.7222 | 0.7152 |
| Nimitz                 | 19  | 32   | 0.9474 | 0.8208 |
| Midway                 | 5   | 13   | 1.0000 | 0.7682 |
| Ticonderoga            | 77  | 92   | 0.8831 | 0.8182 |
| Other-Destroyer        | 46  | 177  | 0.8913 | 0.3533 |
| Atago-DD               | 41  | 132  | 0.9756 | 0.6032 |
| Arleigh-Burke-DD       | 130 | 168  | 0.9769 | 0.8910 |
| Hatsuyuki-DD           | 22  | 80   | 0.8182 | 0.4810 |
| Hyuga-DD               | 16  | 22   | 1.0000 | 0.9899 |
| Asagiri-DD             | 14  | 68   | 0.5000 | 0.2008 |
| Other-Frigate          | 28  | 106  | 0.6786 | 0.4455 |
| Perry-FF               | 77  | 117  | 0.9481 | 0.8748 |
| Patrol                 | 37  | 56   | 0.6486 | 0.4464 |
| Other-Landing          | 18  | 39   | 0.2778 | 0.1991 |
| YuTing-LL              | 19  | 32   | 0.8947 | 0.6619 |
| YuDeng-LL              | 11  | 12   | 0.3636 | 0.2603 |
| YuDao-LL               | 5   | 15   | 1.0000 | 0.8474 |
| YuZhao-LL              | 6   | 22   | 1.0000 | 0.8000 |
| Austin-LL              | 27  | 48   | 0.9259 | 0.8042 |
| Osumi-LL               | 6   | 14   | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 10   | 0.8333 | 0.8182 |
| LSD-41-LL              | 21  | 63   | 0.9048 | 0.8453 |
| LHA-LL                 | 31  | 48   | 0.9677 | 0.9028 |
| Commander              | 32  | 67   | 0.9375 | 0.8617 |
| Other-Auxiliary-Ship   | 18  | 67   | 0.5000 | 0.2856 |
| Medical-Ship           | 5   | 19   | 0.8000 | 0.7636 |
| Test-Ship              | 7   | 23   | 0.7143 | 0.5379 |
| Training-Ship          | 11  | 36   | 1.0000 | 0.9341 |
| AOE                    | 11  | 42   | 0.9091 | 0.5061 |
| Masyuu-AS              | 8   | 33   | 1.0000 | 0.6997 |
| Sanantonio-AS          | 13  | 33   | 0.8462 | 0.7778 |
| EPF                    | 10  | 13   | 0.8000 | 0.8182 |
| Other-Merchant         | 50  | 81   | 0.2600 | 0.0965 |
| Container-Ship         | 72  | 122  | 0.7500 | 0.5340 |
| RoRo                   | 20  | 37   | 0.9000 | 0.8710 |
| Cargo                  | 169 | 271  | 0.8284 | 0.6700 |
| Barge                  | 22  | 60   | 0.2727 | 0.2102 |
| Tugboat                | 46  | 82   | 0.6957 | 0.4934 |
| Ferry                  | 53  | 116  | 0.5660 | 0.4327 |
| Yacht                  | 140 | 208  | 0.8929 | 0.7261 |
| Sailboat               | 341 | 67   | 0.1466 | 0.1608 |
| Fishing-Vessel         | 99  | 225  | 0.6061 | 0.2878 |
| Oil-Tanker             | 32  | 64   | 0.6875 | 0.5392 |
| Hovercraft             | 31  | 62   | 0.8387 | 0.5866 |
| Motorboat              | 398 | 487  | 0.3442 | 0.1966 |
| Dock                   | 154 | 158  | 0.4805 | 0.4127 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.6102 |
+------------------------+-----+------+--------+--------+

cls

+------------------------+-----+------+--------+--------+
| class                  | gts | dets | recall | ap     |
+------------------------+-----+------+--------+--------+
| Other-Ship             | 297 | 755  | 0.5758 | 0.3391 |
| Other-Warship          | 209 | 599  | 0.7368 | 0.4678 |
| Submarine              | 171 | 221  | 0.8304 | 0.7925 |
| Other-Aircraft-Carrier | 4   | 30   | 1.0000 | 1.0000 |
| Enterprise             | 18  | 44   | 0.7222 | 0.6950 |
| Nimitz                 | 19  | 44   | 1.0000 | 0.8555 |
| Midway                 | 5   | 28   | 1.0000 | 0.6591 |
| Ticonderoga            | 77  | 117  | 0.8961 | 0.8153 |
| Other-Destroyer        | 46  | 251  | 0.9565 | 0.3727 |
| Atago-DD               | 41  | 162  | 0.9756 | 0.5623 |
| Arleigh-Burke-DD       | 130 | 236  | 0.9692 | 0.8918 |
| Hatsuyuki-DD           | 22  | 169  | 1.0000 | 0.4221 |
| Hyuga-DD               | 16  | 35   | 1.0000 | 0.9893 |
| Asagiri-DD             | 14  | 129  | 0.7857 | 0.2912 |
| Other-Frigate          | 28  | 225  | 0.7143 | 0.4763 |
| Perry-FF               | 77  | 171  | 0.9481 | 0.8799 |
| Patrol                 | 37  | 139  | 0.6486 | 0.4139 |
| Other-Landing          | 18  | 106  | 0.6667 | 0.2403 |
| YuTing-LL              | 19  | 44   | 0.8947 | 0.6636 |
| YuDeng-LL              | 11  | 45   | 0.3636 | 0.2132 |
| YuDao-LL               | 5   | 37   | 1.0000 | 0.8474 |
| YuZhao-LL              | 6   | 50   | 1.0000 | 0.6946 |
| Austin-LL              | 27  | 78   | 0.9259 | 0.7507 |
| Osumi-LL               | 6   | 42   | 1.0000 | 1.0000 |
| Wasp-LL                | 6   | 39   | 1.0000 | 0.9545 |
| LSD-41-LL              | 21  | 102  | 0.9524 | 0.8407 |
| LHA-LL                 | 31  | 74   | 0.9677 | 0.9002 |
| Commander              | 32  | 108  | 1.0000 | 0.8990 |
| Other-Auxiliary-Ship   | 18  | 133  | 0.6667 | 0.3216 |
| Medical-Ship           | 5   | 57   | 0.8000 | 0.6667 |
| Test-Ship              | 7   | 62   | 0.7143 | 0.4545 |
| Training-Ship          | 11  | 57   | 1.0000 | 0.8885 |
| AOE                    | 11  | 75   | 0.9091 | 0.5703 |
| Masyuu-AS              | 8   | 61   | 1.0000 | 0.5794 |
| Sanantonio-AS          | 13  | 75   | 0.9231 | 0.8093 |
| EPF                    | 10  | 38   | 0.8000 | 0.8182 |
| Other-Merchant         | 50  | 160  | 0.3400 | 0.1629 |
| Container-Ship         | 72  | 211  | 0.8750 | 0.5725 |
| RoRo                   | 20  | 56   | 0.9500 | 0.8782 |
| Cargo                  | 169 | 345  | 0.8757 | 0.6897 |
| Barge                  | 22  | 132  | 0.3636 | 0.2082 |
| Tugboat                | 46  | 135  | 0.7609 | 0.5428 |
| Ferry                  | 53  | 231  | 0.6415 | 0.4688 |
| Yacht                  | 140 | 282  | 0.9214 | 0.7681 |
| Sailboat               | 341 | 212  | 0.3343 | 0.2737 |
| Fishing-Vessel         | 99  | 437  | 0.6768 | 0.2626 |
| Oil-Tanker             | 32  | 120  | 0.7812 | 0.6008 |
| Hovercraft             | 31  | 74   | 0.8387 | 0.5004 |
| Motorboat              | 398 | 715  | 0.3894 | 0.1951 |
| Dock                   | 154 | 306  | 0.5260 | 0.4598 |
+------------------------+-----+------+--------+--------+
| mAP                    |     |      |        | 0.6124 |
+------------------------+-----+------+--------+--------+
cls_top'''
'''v701的epoch36
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
# os.system("CUDA_VISIBLE_DEVICES=0,1 bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/epoch_36.pth \
#            2\
#            --eval mAP")

# os.system("bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v701_faster_rcnn_roitrans_r50_fpn_PCB_clsscore_weight/results-36/")

# os.system("bash /home/guobo/OBBDetection/tools/dist_test.sh \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight.py \
#            /home/guobo/OBBDetection/TrainTest/work_dirs/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight/epoch_36.pth \
#            4\
#            --format-only \
#            --options \
#            save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v801_faster_rcnn_roitrans_r50_fpn_PCB_loss_weight/results-36/"
#           )

'''v008-RoITrans-68-epoch'''
os.system("bash /home/guobo/OBBDetection/tools/dist_test.sh \
           /home/guobo/OBBDetection/TrainTest/work_dirs/v008_faster_rcnn_roitrans_r50_fpn_nopretrain/v008_faster_rcnn_roitrans_r50_fpn_nopretrain.py \
           /home/guobo/OBBDetection/TrainTest/work_dirs/v008_faster_rcnn_roitrans_r50_fpn_nopretrain/epoch_68.pth \
           4\
           --format-only \
           --options \
           save_dir=/home/guobo/OBBDetection/TrainTest/work_dirs/v008_faster_rcnn_roitrans_r50_fpn_nopretrain/results-68/"
          )
