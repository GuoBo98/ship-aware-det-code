import os
import cv2
from tqdm import trange
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def parse_obj(xml_path, filename):

    tree = ET.parse(xml_path + filename + '.xml')
    objects = []

    for obj in tree.findall('object'):

        name = obj.find('name').text
        robndBox = obj.find('robndbox')
        cx = float(robndBox.find('cx').text)
        cy = float(robndBox.find('cy').text)
        w = float(robndBox.find('w').text)
        h = float(robndBox.find('h').text)
        angle = float(robndBox.find('angle').text)
        name_id = categories_dict[name]
        bbox = [cx, cy, w, h, angle, name_id]

        objects.append(bbox)

    return objects


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


def plot_confusion_matrix(
    cm,
    classes,
    save_img_path,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
):
    fig, ax = plt.subplots(figsize=(20, 20), dpi=200)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_img_path)


if __name__ == '__main__':

    gtPath = "/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/val_set/xmls/"
    work_dirs = '/data2/guobo/01_SHIPRSDET/TrainTestv2/work_dirs/v1501_ShipIS_GRoI_RRoIAlign_75/'
    resultPath = os.path.join(work_dirs, 'results-76-voc/')
    save_img_path = os.path.join(work_dirs, 'confusion_matrix_76.png')
    save_img_path_norm = os.path.join(work_dirs,
                                      'confusion_matrix_76_norm.png')

    # resultPath = os.path.join(work_dirs, 'results-36-cross-voc/')
    # save_img_path = os.path.join(work_dirs, 'confusion_matrix_cross-36.png')
    # save_img_path_norm = os.path.join(work_dirs,
    #                                   'confusion_matrix_cross-36_norm.png')

    categories = [
        'Other-Ship', 'Other-Warship', 'Submarine', 'Other-Aircraft-Carrier',
        'Enterprise', 'Nimitz', 'Midway', 'Ticonderoga', 'Other-Destroyer',
        'Atago-DD', 'Arleigh-Burke-DD', 'Hatsuyuki-DD', 'Hyuga-DD',
        'Asagiri-DD', 'Other-Frigate', 'Perry-FF', 'Patrol', 'Other-Landing',
        'YuTing-LL', 'YuDeng-LL', 'YuDao-LL', 'YuZhao-LL', 'Austin-LL',
        'Osumi-LL', 'Wasp-LL', 'LSD-41-LL', 'LHA-LL', 'Commander',
        'Other-Auxiliary-Ship', 'Medical-Ship', 'Test-Ship', 'Training-Ship',
        'AOE', 'Masyuu-AS', 'Sanantonio-AS', 'EPF', 'Other-Merchant',
        'Container-Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry',
        'Yacht', 'Sailboat', 'Fishing-Vessel', 'Oil-Tanker', 'Hovercraft',
        'Motorboat', 'Dock', 'background'
    ]
    categories_id = [i for i in range(50)]
    categories_dict = dict(zip(categories, categories_id))
    categories_dict_reverse = dict(
        zip(categories_dict.values(), categories_dict.keys()))

    gt_list = os.listdir(gtPath)
    result_list = os.listdir(resultPath)

    confusion_matrix = np.zeros((51, 51))
    confusion_matrix_norm = np.zeros((51, 51))
    for i in trange(len(gt_list)):
        xml_name = gt_list[i][0:-4]
        gt_bboxs = parse_obj(gtPath, xml_name)
        result_bboxs = parse_obj(resultPath, xml_name)
        for gt_bbox in gt_bboxs:
            gt_bbox_pointobb = thetaobb2pointobb(gt_bbox)
            for result_bbox in result_bboxs:
                result_bboxs_pointobb = thetaobb2pointobb(result_bbox)
                iou = theta_iou(result_bboxs_pointobb, gt_bbox_pointobb)
                '''gt_bbox[5]指的是box框的类别id'''
                if iou > 0.5:
                    gt_bbox.append('flag')
                    result_bbox.append('flag')
                    confusion_matrix[gt_bbox[5],
                                     result_bbox[5]] = confusion_matrix[
                                         gt_bbox[5], result_bbox[5]] + 1
        for gt_bbox in gt_bboxs:
            if len(gt_bbox) == 6:
                confusion_matrix[gt_bbox[5],
                                 50] = confusion_matrix[gt_bbox[5], 50] + 1
        for result_bbox in result_bboxs:
            if len(result_bbox) == 6:
                confusion_matrix[50,
                                 gt_bbox[5]] = confusion_matrix[50,
                                                                gt_bbox[5]] + 1

    for i in range(len(confusion_matrix)):
        confusion_matrix_norm[i] = confusion_matrix[i] / np.sum(
            confusion_matrix[i])

    plot_confusion_matrix(confusion_matrix, categories, save_img_path)
    plot_confusion_matrix(confusion_matrix_norm * 100, categories,
                          save_img_path_norm)
