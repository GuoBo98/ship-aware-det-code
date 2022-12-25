import os
import json
from typing_extensions import final
import cv2
import numpy as np
from tqdm import trange
from pycocotools import mask as coco_mask
import mmcv
from shapely.geometry import Polygon


def pointobb2mask(pointobbs):
    cocoMaskList = []
    for i in trange(len(pointobbs)):
        pointbbx = pointobbs[i]
        if pointbbx['score'] > 0:
            newsegm = {}
            newsegm['image_id'] = pointbbx['image_id']
            newsegm['category_id'] = pointbbx['category_id']
            newsegm['score'] = pointbbx['score']
            '''generate the canvas and write mask on it, then transform the canvas to cocoMask'''
            # canvas = np.zeros((int(pointbbx['image_width']),int(pointbbx['image_height'])))
            canvas = np.zeros(
                (int(pointbbx['image_height']), int(pointbbx['image_width'])))
            point = pointbbx['pointobbs']
            point = np.reshape(point, (4, 2)).astype(int)
            cv2.fillPoly(canvas, [point], (255, 255, 255))
            canvas = np.asfortranarray(canvas)
            cocoMask = coco_mask.encode(canvas.astype(np.uint8))
            '''the type of cocoMask['counts] is byte which is can't be dumped into json'''
            cocoMask['counts'] = str(cocoMask['counts'], 'utf-8')
            newsegm['segmentation'] = cocoMask
            cocoMaskList.append(newsegm)
    return cocoMaskList


def all_roNMS_cross(all_objects):
    iou_th = 0.3
    tmp_objects = []
    re_idx = []
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['pointobbs']
        score1 = obj['score']

        for idx_c in range(idx + 1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            if obj_c['category_id'] == obj['category_id']:
                bbox2 = obj_c['pointobbs']
                score2 = obj_c['score']
                iou, inter, area1, area2 = theta_iou(bbox1, bbox2)
                ############--------############
                if iou > iou_th:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
                elif inter == area1 or inter == area2:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
            else:
                '''类间nms也要做'''
                bbox2 = obj_c['pointobbs']
                score2 = obj_c['score']
                iou, inter, area1, area2 = theta_iou(bbox1, bbox2)
                ############--------############
                if iou > iou_th:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
                elif inter == area1 or inter == area2:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
        if idx not in re_idx:
            tmp_objects.append(obj)

    return tmp_objects


def all_roNMS(all_objects):
    iou_th = 0.3
    tmp_objects = []
    re_idx = []
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['pointobbs']
        score1 = obj['score']

        for idx_c in range(idx + 1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            if obj_c['category_id'] == obj['category_id']:
                bbox2 = obj_c['pointobbs']
                score2 = obj_c['score']
                iou, inter, area1, area2 = theta_iou(bbox1, bbox2)
                ############--------############
                if iou > iou_th:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
                elif inter == area1 or inter == area2:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
        if idx not in re_idx:
            tmp_objects.append(obj)

    return tmp_objects


def simple_robb_xml_dump(objects, img_name, save_file_path):
    bboxes, pointobbs, labels, scores, rbbox, num = [], [], [], [], [], 0
    for obj in objects:
        pointobbs.append(obj['pointobbs'])
        rbbox.append(pointobb2thetaobb(obj['pointobbs']))
        labels.append(obj['categories_name'])
        scores.append(obj['score'])
        num += 1

    xml_file = open(save_file_path, 'w', encoding='utf-8')
    # xml_file.write('<?xml version=' + '\"1.0\"' + ' encoding='+ 'utf-8' + '?>\n')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('	<filename>' + str(img_name) + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + 'Unknown' + '</width>\n')
    xml_file.write('        <height>' + 'Unknown' + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for idx in range(num):

        name = labels[idx]
        sco = str(scores[idx])
        cx, cy, w, h, angle = rbbox[idx]
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + name + '</name>\n')
        xml_file.write('		<probability>' + sco + '</probability>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <type>robndbox</type>\n')
        xml_file.write('        <robndbox>\n')
        xml_file.write('            <cx>' + str(cx) + '</cx>\n')
        xml_file.write('            <cy>' + str(cy) + '</cy>\n')
        xml_file.write('            <w>' + str(w) + '</w>\n')
        xml_file.write('            <h>' + str(h) + '</h>\n')
        xml_file.write('            <angle>' + str(angle) + '</angle>\n')
        xml_file.write('        </robndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')


def pointobb2thetaobb(pointobb):
    """convert pointobb to thetaobb
    Input:
        pointobb (list[1x8]): [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        thetaobb (list[1x5])
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]

    return thetaobb


def theta_iou(pointobb1, pointobb2):
    """
    docstring here
        :param pointobb1, pointobb2 : list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return: iou, inter, area1, area2
    """
    pointobb1 = np.asarray(pointobb1)
    pointobb2 = np.asarray(pointobb2)
    pointobb1 = Polygon(pointobb1[:8].reshape((4, 2)))
    pointobb2 = Polygon(pointobb2[:8].reshape((4, 2)))
    if not pointobb1.is_valid or not pointobb2.is_valid:
        return 0, 0, 0, 0
    inter = Polygon(pointobb1).intersection(Polygon(pointobb2)).area
    union = pointobb1.area + pointobb2.area - inter
    if union == 0:
        return 0, 0, pointobb1.area, pointobb2.area
    else:
        return inter / union, inter, pointobb1.area, pointobb2.area


if __name__ == '__main__':
    work_dirs = '/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1602_faster_rcnn_orpn_r50_fpn_IS5/'
    results_file = 'results-93'
    dota_result_path = os.path.join(work_dirs, results_file)
    voc_result_path = os.path.join(work_dirs, results_file + '-voc')
    voc_result_cross_path = os.path.join(work_dirs,
                                         results_file + '-cross-voc')
    if not os.path.exists(voc_result_path):
        os.makedirs(voc_result_path)

    if not os.path.exists(voc_result_cross_path):
        os.makedirs(voc_result_cross_path)

    dota_result_list = os.listdir(dota_result_path)
    bbox_result_list = []
    '''For image name map to image id'''
    # gt_path = '/data2/guobo/01_SHIPRSDET/new-COCO-Format/annotations/shipRS_valset.json'
    gt_path = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Coco/annotations/shipRS_valset.json"

    with open(gt_path, 'r') as f:
        gt = json.load(f)

    image_info = gt['images']
    image_name_map = {}
    for i in range(len(image_info)):
        image_name_map[image_info[i]['file_name']] = i + 1

    new_image_name_map = {v: k for k, v in image_name_map.items()}
    '''For class name map to class id'''

    categories_info = gt['categories']
    categories_name_map = {}
    for i in range(len(categories_info)):
        categories_name_map[categories_info[i]['name']] = i + 1
    '''transform the txt to a list'''
    for i in range(len(dota_result_list)):
        with open(os.path.join(dota_result_path, dota_result_list[i]),
                  'r') as f:
            '''Task1_class to class'''
            class_name = dota_result_list[i].split('.')[0][6:]
            for line in f.readlines():
                line = line.strip('\n')
                line_list = line.split(' ')
                line_list.append(class_name)
                bbox_result_list.append(line_list)

    all_objects, all_objects_nms, all_objects_nms_cross = [], [], []

    for i in trange(550):
        objects = []
        for bbox in bbox_result_list:
            obj = {}
            image_name = bbox[0] + '.bmp'
            image_file_name = new_image_name_map[i + 1]
            if image_name_map[image_name] == i + 1:
                score = bbox[1]
                pointobb = list(map(float, bbox[2:10]))
                categories_name = bbox[10]
                obj['image_id'] = image_name_map[image_name]
                obj['image_name'] = image_name
                obj['pointobbs'] = pointobb
                obj['score'] = float(score)
                obj['categories_name'] = categories_name
                obj['category_id'] = categories_name_map[categories_name]
                obj['image_height'] = gt['images'][image_name_map[image_name] -
                                                   1]['height']
                obj['image_width'] = gt['images'][image_name_map[image_name] -
                                                  1]['width']
                '''设置一个置信度阈值为0.1'''
                if obj['score'] > 0.1:
                    objects.append(obj)
        objects_nms = all_roNMS(objects)
        objects_nms_cross = all_roNMS_cross(objects)
        simple_robb_xml_dump(
            objects_nms, image_file_name,
            os.path.join(voc_result_path, image_file_name[0:-4] + '.xml'))
        simple_robb_xml_dump(
            objects_nms_cross, image_file_name,
            os.path.join(voc_result_cross_path,
                         image_file_name[0:-4] + '.xml'))
        all_objects.append(objects)
        all_objects_nms.append(objects_nms)
        all_objects_nms_cross.append(objects_nms_cross)

    final_objects_nms = []
    for objects in all_objects_nms:
        for obj in objects:
            final_objects_nms.append(obj)

    final_objects_nms_cross = []
    for objects in all_objects_nms_cross:
        for obj in objects:
            final_objects_nms_cross.append(obj)

    final_objects = []
    for objects in all_objects:
        for obj in objects:
            final_objects.append(obj)

    final_objects_nms_mask = pointobb2mask(final_objects_nms)
    final_objects_nms_cross_mask = pointobb2mask(final_objects_nms_cross)

    mmcv.dump(final_objects_nms_mask,
              os.path.join(work_dirs, results_file + '-nms-mask.json'))

    mmcv.dump(final_objects_nms_cross_mask,
              os.path.join(work_dirs, results_file + '-nms-cross-mask.json'))
