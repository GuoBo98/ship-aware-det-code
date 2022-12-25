from mmdet.datasets import PIPELINES
import json
import os
import cv2
import tqdm
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors
from copy import copy


@PIPELINES.register_module()
class InstanceSwitch2:

    def get_curvature(self, obj_crop):
        a = max(obj_crop.shape[0], obj_crop.shape[1])
        b = min(obj_crop.shape[0], obj_crop.shape[1])
        curvature = (a / b )
        return float(curvature)

    def get_bg_hsv_avg(self, obj_crop):
        obj_hsv = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2HSV)[:, [0, -1], :]
        obj_hsv = obj_hsv.reshape((-1, 3))
        bg_hsv = np.average(obj_hsv, axis=0)
        return bg_hsv

    def __init__(
        self,
        data_root="/data2/guobo/01_SHIPRSDET/new-COCO-Format/",
        switch_prob=0.5,
        ann_path="/data2/guobo/01_SHIPRSDET/new-COCO-Format/annotations/shipRS_trainset.json"
    ):
        self.switch_prob = switch_prob
        crop_instance_path = os.path.join(data_root, "train_crop")
        crop_list = os.listdir(crop_instance_path)

        ann_file = open(ann_path)
        ann_json = json.load(ann_file)
        annotations = ann_json["annotations"]
        num_bridges = len(annotations)
        images = ann_json["images"]

        img_to_instance = {}
        img_to_instance_wo_name = []

        #感兴趣的类别放在这个列表里,只对感兴趣的类别执行InstanceSwitch
        attention_category_list = [
            4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 33, 34, 35
        ]

        print("Registering image to plane...")
        for i in tqdm.trange(num_bridges):
            instance = annotations[i]
            image_id = instance["image_id"]
            image_name = images[image_id -
                                1]["file_name"].split('.')[0] + '.png'
            # print(image_name)
            if image_name not in img_to_instance.keys():
                img_to_instance[image_name] = []
            img_to_instance[image_name].append([
                instance["category_id"], instance["id"], instance["pointobb"]
            ])  ##### mark each plane
            img_to_instance_wo_name.append([
                instance["category_id"], instance["id"], instance["pointobb"]
            ])

        print("Counting the average number of each class...")
        list_class_num = np.zeros(50)
        for image in img_to_instance.items():
            # print(image)
            planes = image[1]
            for plane in planes:
                # print(plane)
                cat_id = plane[0]
                list_class_num[
                    cat_id -
                    1] += 1  ##### cat_id从1开始 1~50 但list_calss_num中存储的cat_id从0 开始

        attention_category_sum = 0
        for i in range(len(list_class_num)):
            if i + 1 in attention_category_list:
                attention_category_sum += list_class_num[i]

        self.avg_class_num = math.floor(attention_category_sum /
                                        len(attention_category_list))

        # select the cat_id that need to be replaced
        print("Selectiong the cat_id needs to be replaced...")
        self.switch_class = [
        ]  ##### list of cat_id with numbers less than avg,是从1开始结算的
        self.more_class = []  ##### list of cat_id with numbers more than avg
        self.switch_num_of_less = [
        ]  ##### list of the number of planes needs to replace the larger
        self.switch_num_of_more = [
        ]  ##### list of the number of planes needs to be replaced
        for i in range(len(list_class_num)):
            if list_class_num[
                    i] < self.avg_class_num and i + 1 in attention_category_list:
                self.switch_num_of_less.append(self.avg_class_num -
                                               list_class_num[i])
                self.switch_class.append(
                    i + 1)  ##### switch_class 中存储的cat_id 从1 开始 1~10
            if list_class_num[
                    i] > self.avg_class_num and i + 1 in attention_category_list:
                self.switch_num_of_more.append(list_class_num[i] -
                                               self.avg_class_num)
                self.more_class.append(i + 1)

        print("Registering cropped instances...")
        self.crop_bridge = []
        crop_args = []
        for i in tqdm.trange(len(crop_list)):
            crop_filename = crop_list[i]
            cat_id = int(crop_filename.split('-')[0])
            if cat_id not in self.switch_class:
                continue
            obj_crop = cv2.imread(
                os.path.join(crop_instance_path, crop_filename))
            if obj_crop.shape[0] * obj_crop.shape[1] < 1500:
                continue
            # bg_hsv = self.get_bg_hsv_avg(obj_crop) / 30
            curvature = self.get_curvature(obj_crop)
            self.crop_bridge.append([
                cat_id, obj_crop
            ])  ##### to store the cat_id of cropped instances,这是从1开始的
            #考虑到曲率（反应头尖还是不尖）和尺寸大小来进行替换,曲率代表形状 这部分是更重要的
            crop_args.append([
                curvature * 5,
                np.log10(obj_crop.shape[0] * obj_crop.shape[1]) * 3
            ])
        # KNN fitting
        print("Building a tree...")
        self.crop_bridge = np.asarray(self.crop_bridge, dtype=object)
        crop_args = np.asarray(crop_args)
        self.crop_nbrs = NearestNeighbors(
            n_neighbors=10, algorithm='ball_tree').fit(crop_args)
        self.attention_category = attention_category_list
        print('Complete registration!')

    def get_distence(self, x, y):
        return (abs(x[0] - y[0])**2 + abs(x[1] - y[1])**2)**0.5

    def __call__(self, results):
        if random.random() > self.switch_prob:
            return results
        img_cv = results["img"]
        origin_img = copy(img_cv)
        filename = results["ori_filename"]
        ann_dict = results["ann_info"]
        #读入bboxes，其实这个就是masks
        bridges = ann_dict["bboxes"]
        ori_cat_ids = ann_dict["labels"]
        # print(ori_cat_ids)
        # print(len(bridges))
        # print(img_cv.shape)
        # assert False
        # print(results)
        # assert False
        # print(ann_dict)
        # assert False
        switch_vis_path = "/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_vis/"
        # cv2.imwrite(os.path.join(switch_vis_path,filename),img_cv)

        ##### 如何获取一个飞机样本的矩形框和类别信息 bridge:矩形框坐标， ori_cat_id:类别

        for ii, bridge_ in enumerate(bridges):
            kk = random.random()
            if kk > self.switch_prob:
                # print(kk)
                continue
            # 在IS中，类别是从1开始的，数据集中类别是从0开始的，这里为了匹配给类别加一个1
            ori_cat_id = ori_cat_ids[
                ii] + 1  ##### +1是因为 ann_dict["labels"] 中存储的类别序号从0开始 即0~9

            if ori_cat_id not in self.attention_category:  ##如果不在感兴趣的类里，就跳过
                continue
            if ori_cat_id in self.switch_class:  ##### 若该instance的类型是小类，则跳过
                continue
            if self.switch_num_of_more[self.more_class.index(
                    ori_cat_id)] == 0:  ####如果在大类且大类需要替换的目标数减为0，则跳过
                continue

            # bridge = bridge_[0]
            #bridge是目标框的坐标
            bridge = bridge_
            #canvas生成一个1024*1024的幕布
            canvas = np.zeros_like(img_cv[:, :, 0])
            #把目标框转成4*2的形式
            obb = np.array(bridge, np.int32).reshape((4, 1, 2))
            #生成obb的最小外接矩形，opencv定义法，角度是-90到0度
            rect = cv2.minAreaRect(obb)
            ((cx, cy), (w, h), theta) = rect
            # 这个scale_factor是用来干什么的？使得短边扩张一点,应用于桥
            # if w > h:
            #     rect = ((cx, cy), (w * 1.2, h), theta)
            # else:
            #     rect = ((cx, cy), (w, h * 1.2), theta)
            rect = ((cx, cy), (w, h), theta)
            #又转换成了四个点
            box = cv2.boxPoints(rect)
            box = np.asarray(box).astype(np.int32)
            #canvas相当于是原图舰船的mask
            canvas = cv2.fillPoly(canvas, [box], 255)

            #dst_pts是原图中目标的坐标
            dst_pts = box.astype("float32")
            '''为啥要warp into 100*100来计算背景值'''
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(
                    dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array(
                    [[0 + 1, 100 - 1 - 1], [0 + 1, 0 + 1],
                     [100 - 1 - 1, 0 + 1], [100 - 1 - 1, 100 - 1 - 1]],
                    dtype="float32")
            else:
                src_pts = np.array(
                    [[0 + 1, 0 + 1], [100 - 1 - 1, 0 + 1],
                     [100 - 1 - 1, 100 - 1 - 1], [0 + 1, 100 - 1 - 1]],
                    dtype="float32")

            M0 = cv2.getPerspectiveTransform(dst_pts, src_pts)
            origin_bridge_0 = cv2.warpPerspective(img_cv, M0, (100, 100))
            #计算曲率
            orig_curvature = float(max(w, h) / min(w, h) * 5)

            # orig_bg_hsv = self.get_bg_hsv_avg(origin_bridge_0) / 30

            # Use KNN to get 50 similar bridges and get a random choice.
            find = 0  ##### 用来标记是否找到合适的替换样本
            _, indices = self.crop_nbrs.kneighbors(
                np.array([[orig_curvature,
                           np.log10(w * h) * 3]]))  ##### * 3 original
            for i in range(len(indices[0])):
                #这里修改为随机挑选
                j = random.randint(0,len(indices[0])-1)
                cat_id = self.crop_bridge[indices[0][j]][0]

                if self.switch_num_of_less[self.switch_class.index(
                        cat_id)] == 0:
                    continue
                else:
                    switch_target = self.crop_bridge[indices[0][j]][
                        1]  ##### record cat_id and croped plane data of the nearest available plane
                    # cv2.imwrite(os.path.join(switch_vis_path,filename[0:-4] + '-switch_target' + '.png'),switch_target)
                    target_id = cat_id
                    self.switch_num_of_more[self.more_class.index(
                        ori_cat_id
                    )] -= 1  ##### 更新该大类还需要被替换的个数  放在这里是因为有可能某一个法雷样本找不到任何一个合适的替换样本
                    # print(self.switch_num_of_more)
                    self.switch_num_of_less[self.switch_class.index(
                        cat_id)] -= 1  ##### 更新该小类还需要替换的个数
                    # print(self.switch_num_of_less)
                    find = 1
                    break

            if find == 0:
                continue

            switch_h = switch_target.shape[0]
            switch_w = switch_target.shape[1]

            # To align the long side of instance crop.
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(
                    dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array([[0 + 1, switch_h - 1 - 1], [0 + 1, 0 + 1],
                                    [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1]],
                                   dtype="float32")
            else:
                src_pts = np.array([[0 + 1, 0 + 1], [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1],
                                    [0 + 1, switch_h - 1 - 1]],
                                   dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M_inv = np.linalg.inv(M)

            # Get cropped original bridge.
            origin_bridge = cv2.warpPerspective(img_cv, M_inv,
                                                (switch_w, switch_h))
            # cv2.imwrite(os.path.join(switch_vis_path,filename[0:-4] + '-origin_bridge' + '.png'),origin_bridge)

            # Generate gaussian kernel.
            '''
            gaussian_kernel = cv2.getGaussianKernel(switch_w, switch_w / 7).reshape((switch_w,))
            gaussian_kernel /= gaussian_kernel.max()
            mask_weight = gaussian_kernel * np.ones((switch_h, switch_w))
            mask_weight_h = mask_weight.T
            mask_weight_h = mask_weight_h[:, :, None]
            mask_weight_h = np.concatenate([mask_weight_h, mask_weight_h, mask_weight_h], axis=-1)
            mask_weight = mask_weight[:, :, None]
            mask_weight = np.concatenate([mask_weight, mask_weight, mask_weight], axis=-1)

            mask_weight_h = cv2.resize(mask_weight_h, (switch_w, switch_h))
            mask_weight = mask_weight + mask_weight_h
            mask_weight[mask_weight > 0.8] = 1

            mask_weight_inv = 1 - mask_weight
            # Fuse instance.
            switch_target = switch_target * mask_weight + origin_bridge * mask_weight_inv
            '''
            '''融合这部分还需要改一下'''
            '''
            gaussian_kernel = cv2.getGaussianKernel(switch_w, switch_w / 7).reshape((switch_w,))
            gaussian_kernel /= gaussian_kernel.max()
            mask_weight = gaussian_kernel * np.ones((switch_h, switch_w))
            mask_weight = mask_weight[:, :, None]
            mask_weight = np.concatenate([mask_weight, mask_weight, mask_weight], axis=-1)
            mask_weight[mask_weight > 0.8] = 1
            
            mask_weight_inv = 1 - mask_weight
            
            switch_target = switch_target * mask_weight_inv + origin_bridge * mask_weight
            # switch_target_fuse1 = switch_target * mask_weight_inv + origin_bridge * mask_weight
            '''

            cv2.imwrite(os.path.join('/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_test/',filename[0:-4] + '-img' + '.png'),img_cv)
            cv2.imwrite(os.path.join('/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_test/',filename[0:-4] + '-origin_bridge' + '.png'),origin_bridge)
            cv2.imwrite(os.path.join('/data2/guobo/01_SHIPRSDET/dota-format/split_ss_1024/switch_test/',filename[0:-4] + '-switch_target' + '.png'),switch_target)
            
            
            gaussian_kernel = cv2.getGaussianKernel(switch_h,
                                                    switch_h / 7).reshape(
                                                        (switch_h, ))
            gaussian_kernel /= gaussian_kernel.max()
            
            mask_weight = gaussian_kernel * np.ones((switch_w, switch_h))
            mask_weight = mask_weight.T
            mask_weight = mask_weight[:, :, None]
            mask_weight = np.concatenate(
                [mask_weight, mask_weight, mask_weight], axis=-1)
            
            mask_weight[mask_weight > 0.8] = 0.8
            
            # mask_weight /= 0.8

            mask_weight_inv = 1 - mask_weight

            switch_target = switch_target * mask_weight_inv + origin_bridge * mask_weight

            # switch_target = switch_target * mask_weight + origin_bridge * mask_weight_inv
            
            switch_target.astype(np.uint8)

            # Paste instance onto image.
            switch_warped = cv2.warpPerspective(
                switch_target, M, (img_cv.shape[1], img_cv.shape[0]))
            # img_norm = cv2.normalize(img_cv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #
            # img_norm.astype(np.uint8)
            # img_cv = img_norm
            # if num_switch < 10:
            #     cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_ori.png', img_cv)
            img_cv[canvas == 255] = switch_warped[canvas == 255]

            ##### 替换完样本后，如何获取gt_labels并进行修改
            #这里target-1是因为 ann_dict["labels"] 中返回的类别序号是从0开始的
            ori_cat_ids[ii] = target_id - 1  #####

        if not (origin_img == img_cv).all():
            cv2.imwrite(
                os.path.join(switch_vis_path, filename[0:-4] + '.png'),
                origin_img)
            cv2.imwrite(
                os.path.join(switch_vis_path,
                             filename[0:-4] + '-switched' + '.png'), img_cv)

        return results