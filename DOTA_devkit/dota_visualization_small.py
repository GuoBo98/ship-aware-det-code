from copy import copy
import os
import cv2
from tqdm import trange
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon

COLORS = {
    'Red': (75, 25, 230),
    'Yellow': (25, 225, 225),
    'Green': (75, 180, 60),
    'Blue': (200, 130, 0)
}


def show_thetaobb(img, thetaobb, color):
    """show single theteobb

    Args:
        im (np.array): input image
        thetaobb (list): [cx, cy, w, h, theta]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: image with thetaobb
    """

    cx, cy, w, h, theta = thetaobb

    rect = ((cx, cy), (w, h), theta / np.pi * 180.0)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    cv2.drawContours(img, [rect], -1, color, 3)

    return img


def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self:
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(
        ((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]),
         thetaobb[4] * 180.0 / np.pi))
    box = np.reshape(box, [
        -1,
    ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb


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
        return 0
    inter = Polygon(pointobb1).intersection(Polygon(pointobb2)).area
    union = pointobb1.area + pointobb2.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def parse_obj(xml_path, filename):

    tree = ET.parse(xml_path + filename + '.xml')
    objects = []

    for obj in tree.findall('object'):

        name = obj.find('name').text
        robndBox = obj.find('robndbox')
        try:
            score = float(obj.find('probability').text)
        except:
            score = 1
        cx = float(robndBox.find('cx').text)
        cy = float(robndBox.find('cy').text)
        w = float(robndBox.find('w').text)
        h = float(robndBox.find('h').text)
        angle = float(robndBox.find('angle').text)
        bbox = [cx, cy, w, h, angle, name, round(score, 4)]

        objects.append(bbox)

    return objects


def result_visualization(imagePath, gtPath, resultPath, draw_image_save_path):

    imageList = os.listdir(imagePath)
    '''all_TP means the TP of the whole valset'''
    all_TP, all_FP, all_FN, all_FC = [], [], [], []
    attention_class = ['Motorboat', 'Hovercraft', 'Sailboat', 'Yacht', 'Barge', 'RoRo']
    for i in trange(len(imageList)):
        gtShot = []
        img = imageList[i]
        image = cv2.imread(os.path.join(imagePath, img))
        image_ori = copy(image)
        ##FC means false classify
        TP, FP, FN, FC, FC_GT = [], [], [], [], []
        imgName = img[0:-4]
        gtObjects = parse_obj(gtPath, imgName)
        resultObjects = parse_obj(resultPath, imgName)
        for resultBbox in resultObjects:
            flag = 0
            for gtBbox in gtObjects:
                resultBboxPoint = thetaobb2pointobb(resultBbox[0:5])
                gtBboxPoint = thetaobb2pointobb(gtBbox[0:5])
                iou = theta_iou(resultBboxPoint, gtBboxPoint)
                ##shot
                try:
                    if iou > 0.5 and resultBbox[5] == gtBbox[5]:
                        TP.append(resultBbox)
                        gtShot.append(gtBbox)
                        flag = 1

                    if iou > 0.5 and resultBbox[5] != gtBbox[5]:
                        FC.append(resultBbox)
                        FC_GT.append(gtBbox)
                        gtShot.append(gtBbox)
                        flag = 1
                except:
                    print('The iou is', iou)
                ##FN
            if flag == 0:
                FP.append(resultBbox)
        for gtBbox in gtObjects:
            ##FN
            if gtBbox not in gtShot:
                FN.append(gtBbox)

        ##draw bbox on the image
        for bbox in TP:
            if bbox[5] in attention_class:
                '''append the bbox to all_TP for the caculation of precisiaon and recall'''
                all_TP.append(bbox)
                pointobb = thetaobb2pointobb(bbox[0:5])
                image = show_thetaobb(image, bbox[0:5], COLORS['Green'])
                # if len(TP) < 5:
                #     image = cv2.putText(image, str(
                #         bbox[5]), (int(pointobb[0]) + 10, int(pointobb[1]) + 10),
                #                         cv2.FONT_HERSHEY_COMPLEX, 0.5,
                #                         COLORS['Green'], 1)
        for bbox in FP:
            if bbox[5] in attention_class:
                all_FP.append(bbox)
                pointobb = thetaobb2pointobb(bbox[0:5])
                image = show_thetaobb(image, bbox[0:5], COLORS['Blue'])
                # image = cv2.putText(
                #     image,
                #     str(str(bbox[5]) + '-' + str(bbox[6])),
                #     (int(pointobb[0]) + 10, int(pointobb[1]) - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     COLORS['Blue'],
                #     thickness=1)

        for bbox in FC:
            if bbox[5] in attention_class:
                all_FC.append(bbox)
                pointobb = thetaobb2pointobb(bbox[0:5])
                image = show_thetaobb(image, bbox[0:5], COLORS['Yellow'])
                if len(FC) < 5:
                    pass
                    # image = cv2.putText(image, str(bbox[5]),
                    #                     (int(pointobb[0]), int(pointobb[1]) + 70),
                    #                     cv2.FONT_HERSHEY_COMPLEX, 1,
                    #                     COLORS['Yellow'], 1)
                    # image = cv2.putText(image, str(FC_GT[FC.index(bbox)][5]),
                    #                     (int(pointobb[0]), int(pointobb[1]) + 40),
                    #                     cv2.FONT_HERSHEY_COMPLEX, 1,
                    #                     COLORS['Green'], 1)

        for bbox in FN:
            if bbox[5] in attention_class:
                all_FN.append(bbox)
                pointobb = thetaobb2pointobb(bbox[0:5])
                image = show_thetaobb(image, bbox[0:5], COLORS['Red'])
                if len(FN) < 5:
                    pass
                    # image = cv2.putText(image, str(bbox[5]),
                    #                     (int(pointobb[0]), int(pointobb[1]) + 70),
                    #                     cv2.FONT_HERSHEY_COMPLEX, 1, COLORS['Red'],
                    #                     1)
        if not (image == image_ori).all():
            cv2.imwrite(os.path.join(draw_image_save_path, img), image)
    print('The precision is ', (len(all_TP) / (len(all_TP) + len(all_FP))))
    print('The recall is ', ((len(all_TP) + len(all_FC)) /
                             (len(all_TP) + len(all_FC) + len(all_FN))))


if __name__ == '__main__':

    imagePath = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/val_set/images/"
    gtPath = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/val_set/xmls/"

    work_dirs = '/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1702_roitrans_gwd_k10/'
    resultPath = os.path.join(work_dirs, 'results-70-0.05-voc/')
    draw_image_save_path = os.path.join(work_dirs, 'results-70-0.05-show-small/')
    if not os.path.exists(draw_image_save_path):
        os.makedirs(draw_image_save_path)
    result_visualization(imagePath, gtPath, resultPath, draw_image_save_path)