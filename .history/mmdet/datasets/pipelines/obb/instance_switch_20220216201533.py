from mmdet.datasets import PIPELINES
import json
import os
import cv2
import tqdm
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors

@PIPELINES.register_module()
class InstanceSwitch:

    def get_bg_hsv_avg(self, obj_crop):
        obj_hsv = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2HSV)[:, [0, -1], :]
        obj_hsv = obj_hsv.reshape((-1, 3))
        bg_hsv = np.average(obj_hsv, axis=0)
        return bg_hsv

    def __init__(self, data_root="data/GF/airplane/", switch_prob=0.5,ann_path = "data/GF/airplane/ann_new/instances_trainval2021.json"):
        self.switch_prob = switch_prob
        crop_instance_path = os.path.join(data_root,"crop")
        crop_list = os.listdir(crop_instance_path)

        ann_file = open(ann_path)
        ann_json = json.load(ann_file)
        annotations = ann_json["annotations"]
        num_bridges = len(annotations)
        images = ann_json["images"]

        img_to_instance = {}
        img_to_instance_wo_name = []


        print("Registering image to plane...")
        for i in tqdm.trange(num_bridges):
            instance = annotations[i]
            image_id = instance["image_id"]
            image_name = images[image_id - 1]["file_name"].split('.')[0] + '.png'
            # print(image_name)
            if image_name not in img_to_instance.keys():
                img_to_instance[image_name] = []
            img_to_instance[image_name].append(
                [instance["category_id"], instance["id"], instance["pointobb"]])  ##### mark each plane
            img_to_instance_wo_name.append([instance["category_id"], instance["id"], instance["pointobb"]])

        print("Counting the average number of each class...")
        list_class_num = np.zeros(10)
        for image in img_to_instance.items():
            # print(image)
            planes = image[1]
            for plane in planes:
                # print(plane)
                cat_id = plane[0]
                list_class_num[cat_id - 1] += 1  ##### cat_id从1开始 1~10 但list_calss_num中存储的cat_id从0 开始
        self.avg_class_num = math.floor(sum(list_class_num) / 10)
        print("**The number of each class is: {0}".format(list_class_num))
        print("**The average number of all classes is: {}\n".format(self.avg_class_num))

        # select the cat_id that need to be replaced
        print("Selectiong the cat_id needs to be replaced...")
        self.switch_class = []  ##### list of cat_id with numbers less than avg
        self.more_class = []  ##### list of cat_id with numbers more than avg
        self.switch_num_of_less = []  ##### list of the number of planes needs to replace the larger
        self.switch_num_of_more = []  ##### list of the number of planes needs to be replaced
        for i in range(len(list_class_num)):
            if list_class_num[i] < self.avg_class_num:
                self.switch_num_of_less.append(self.avg_class_num - list_class_num[i])
                self.switch_class.append(i + 1)  ##### switch_class 中存储的cat_id 从1 开始 1~10
            if list_class_num[i] > self.avg_class_num:
                self.switch_num_of_more.append(list_class_num[i] - self.avg_class_num)
                self.more_class.append(i + 1)
        # print(switch_class)
        print("**The list of cat_id needs to replace the larger is: {0}".format(self.switch_class))
        print("**The list of cat_id needs to be replaced is: {0}".format(self.more_class))
        print("**The number of planes needs to replace the larger is: {0}".format(self.switch_num_of_less))
        print("**The number of planes needs to be replaced is: {0}\n".format(self.switch_num_of_more))

        print("Registering cropped instances...")
        self.crop_bridge = []
        crop_args = []
        for i in tqdm.trange(len(crop_list)):
            crop_filename = crop_list[i]
            cat_id = int(crop_filename[0]) + 1
            if cat_id not in self.switch_class:
                continue
            obj_crop = cv2.imread(os.path.join(crop_instance_path, crop_filename))
            if obj_crop.shape[0] * obj_crop.shape[1] < 1500:
                continue
            # if obj_crop.shape[0] < obj_crop.shape[1]:
            #     obj_crop = np.rot90(obj_crop)
            bg_hsv = self.get_bg_hsv_avg(obj_crop) / 30
            # km = KMeans(n_clusters=2, random_state=9)
            # _ = km.fit_predict(obj_hsv)
            # cluster_center = km.cluster_centers_
            self.crop_bridge.append([cat_id,obj_crop])            ##### to store the cat_id of cropped instances
            crop_args.append([*bg_hsv, np.log10(obj_crop.shape[0]*obj_crop.shape[1])*3])
        # KNN fitting
        print("Building a tree...")
        self.crop_bridge = np.asarray(self.crop_bridge,dtype=object)
        crop_args = np.asarray(crop_args)
        self.crop_nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(crop_args)

    def get_distence(self, x, y):
        return (abs(x[0] - y[0]) ** 2 + abs(x[1] - y[1]) ** 2) ** 0.5

    def __call__(self, results):
        # if random.random() > self.switch_prob:
        #     return results
        img_cv = results["img"]
        filename = results["ori_filename"]
        ann_dict = results["ann_info"]
        bridges = ann_dict["masks"]
        ori_cat_ids = ann_dict["labels"]
        # print(img_cv.shape)
        # assert False
        # print(results)
        # assert False
        # print(ann_dict)
        # assert False
        switch_vis_path = '/home/zlx/mmdetection/work_dirs/GF-RarePlanes/online_IS_faster_r50_on_newset/switch_vis/'

        ##### 如何获取一个飞机样本的矩形框和类别信息 bridge:矩形框坐标， ori_cat_id:类别
        num_switch = 0

        for ii,bridge_ in enumerate(bridges):
            # print(len(bridges), filename)
            ori_cat_id = ori_cat_ids[ii]+1  ##### +1是因为 ann_dict["labels"] 中存储的类别序号从0开始 即0~9
            if ori_cat_id in self.switch_class:  ##### 若该instance的类型是小类，则跳过
                continue
            if self.switch_num_of_more[self.more_class.index(ori_cat_id)] == 0:
                continue

            bridge = bridge_[0]
            canvas = np.zeros_like(img_cv[:, :, 0])
            obb = np.array(bridge, np.int32).reshape((4, 1, 2))
            rect = cv2.minAreaRect(obb)
            ((cx, cy), (w, h), theta) = rect
            scale_factor = (max(w, h) / min(w, h)) ** 0.5
            if w > h:
                h *= scale_factor
            else:
                w *= scale_factor
            rect = ((cx, cy), (w, h), theta)
            box = cv2.boxPoints(rect)
            box = np.asarray(box).astype(np.int32)
            canvas = cv2.fillPoly(canvas, [box], 255)

            dst_pts = box.astype("float32")

            # Crop and warp the orig bridge into shape 20x20. Get the bg hsv.
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array([[0 + 1, 20 - 1 - 1],
                                    [0 + 1, 0 + 1],
                                    [20 - 1 - 1, 0 + 1],
                                    [20 - 1 - 1, 20 - 1 - 1]], dtype="float32")
            else:
                src_pts = np.array([[0 + 1, 0 + 1],
                                    [20 - 1 - 1, 0 + 1],
                                    [20 - 1 - 1, 20 - 1 - 1],
                                    [0 + 1, 20 - 1 - 1]], dtype="float32")
            M0 = cv2.getPerspectiveTransform(dst_pts, src_pts)
            origin_bridge_0 = cv2.warpPerspective(img_cv, M0, (20, 20))
            orig_bg_hsv = self.get_bg_hsv_avg(origin_bridge_0) / 30

            # Use KNN to get 50 similar bridges and get a random choice.
            find = 0  ##### 用来标记是否找到合适的替换样本
            _, indices = self.crop_nbrs.kneighbors(np.array([[*orig_bg_hsv, np.log10(w * h)*3]])) ##### * 3 original
            for i in range(len(indices[0])):
                cat_id = self.crop_bridge[indices[0][i]][0]
                if self.switch_num_of_less[self.switch_class.index(cat_id)] == 0:
                    continue
                else:
                    switch_target = self.crop_bridge[indices[0][i]][1]  ##### record cat_id and croped plane data of the nearest available plane
                    target_id = cat_id
                    self.switch_num_of_more[self.more_class.index(ori_cat_id)] -= 1  ##### 更新该大类还需要被替换的个数  放在这里是因为有可能某一个法雷样本找不到任何一个合适的替换样本
                    # print(self.switch_num_of_more)
                    self.switch_num_of_less[self.switch_class.index(cat_id)] -= 1  ##### 更新该小类还需要替换的个数
                    # print(self.switch_num_of_less)
                    find = 1
                    break

            if find == 0:
                continue
            switch_h = switch_target.shape[0]
            switch_w = switch_target.shape[1]

            # To align the long side of instance crop.
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array([[0 + 1, switch_h - 1 - 1],
                                    [0 + 1, 0 + 1],
                                    [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1]], dtype="float32")
            else:
                src_pts = np.array([[0 + 1, 0 + 1],
                                    [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1],
                                    [0 + 1, switch_h - 1 - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M_inv = np.linalg.inv(M)

            # Get cropped original bridge.
            origin_bridge = cv2.warpPerspective(img_cv, M_inv, (switch_w, switch_h))
            # print(origin_bridge.shape)
            # print(origin_bridge[:,:,0])
            # Generate gaussian kernel.
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
            switch_target.astype(np.uint8)
            # *****************************************************8*
            if num_switch < 10:
                num_switch += 1
                cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_tar.png', switch_target)

            # Paste instance onto image.
            switch_warped = cv2.warpPerspective(switch_target, M, (img_cv.shape[1], img_cv.shape[0]))
            # img_norm = cv2.normalize(img_cv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #
            # img_norm.astype(np.uint8)
            # img_cv = img_norm
            if num_switch < 10:
                cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_ori.png', img_cv)
            img_cv[canvas == 255] = switch_warped[canvas == 255]
            if num_switch <= 10:
                cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_switched.png', img_cv)
            ##### 替换完样本后，如何获取gt_labels并进行修改
            ori_cat_ids[ii] = target_id-1 ##### -1是因为 ann_dict["labels"] 中返回的类别序号是从0开始的

        ##### print the current distribution of classes
        # present_list = []
        # for i in range(10):
        #     if i + 1 in self.switch_class:
        #         k = self.switch_class.index(i + 1)
        #         present_num = self.avg_class_num - self.switch_num_of_less[k]
        #         present_list.append(present_num)
        #     else:
        #         k = self.more_class.index(i + 1)
        #         present_num = self.switch_num_of_more[k] + self.avg_class_num
        #         present_list.append(present_num)
        # print("####The present number of each class is: {0}".format(present_list))



        return results