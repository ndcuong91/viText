from viText.tools.utils import get_list_file_in_folder
import json, os, cv2, math
import numpy as np


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))


def sort_pts(list_pts):
    points = sorted(list_pts, key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box


def split_receipt_from_raw_img(img_dir, anno_dir, dst_dir):
    '''
    To work with data from labelme
    :param img_dir:
    :param anno_dir:
    :param dst_dir:
    :return:
    '''
    list_img = get_list_file_in_folder(img_dir, ext=['jpg', 'png', 'JPG', 'PNG', 'tif', 'tiff', 'TIFF'])
    list_img = sorted(list_img)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for idx, img_name in enumerate(list_img):
        print(idx, img_name)
        # if idx != 7:
        #     continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        anno_path = os.path.join(anno_dir, img_name.split('.')[0] + '.json')
        if not os.path.exists(anno_path):
            continue
        with open(anno_path) as json_file:
            data = json.load(json_file)
            for idx, poly in enumerate(data['shapes']):
                bbox = poly["points"]
                w = round(euclidean_distance(bbox[0], bbox[1]))
                h = round(euclidean_distance(bbox[1], bbox[2]))
                src_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                target_bbox = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]

                sort_bbox = sort_pts(src_bbox)

                ratio = euclidean_distance(sort_bbox[0], sort_bbox[1]) / w
                if abs(ratio - 1) > 0.05:
                    target_bbox = [[w - 1, 0], [w - 1, h - 1], [0, h - 1], [0, 0]]

                src_pts = np.asarray(sort_bbox, dtype=np.float32)
                dst_pts = np.asarray(target_bbox, dtype=np.float32)
                print(src_pts)
                print(dst_pts)

                perspective_trans, status = cv2.findHomography(src_pts, dst_pts)
                trans_img = cv2.warpPerspective(img, perspective_trans, (w, h),
                                                borderValue=(255, 255, 255))
                #cv2.imshow('res',trans_img)
                cv2.imwrite(os.path.join(dst_dir, img_name.split('.')[0] + '_' + str(idx + 1) + '.jpg'), trans_img)
                cv2.waitKey(0)

import random, shutil
def rename_viReceipts(img_dir, anno_dir, dst_dir):
    prefix='no'
    list_img = get_list_file_in_folder(img_dir)
    random.shuffle(list_img)
    label_dir= os.path.join(dst_dir,'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    for idx, img_name in enumerate(list_img):
        print(idx, img_name)
        base_name = img_name.split('.')[0]
        anno_path_xml = os.path.join(anno_dir, base_name+'.xml')
        new_name = prefix+'_'+str(idx).zfill(4)
        if not os.path.exists(anno_path_xml):
            shutil.copy(os.path.join(img_dir, img_name),os.path.join(dst_dir,new_name+'.jpg'))
        else:
            print('has label!')
            shutil.copy(os.path.join(img_dir, img_name),os.path.join(label_dir,new_name+'.jpg'))
            shutil.copy(anno_path_xml,os.path.join(label_dir, new_name + '.xml'))





if __name__ == "__main__":
    # img_dir = '/home/cuongnd/PycharmProjects/aicr/viText/viText/viData/viReceipts/raw/20210323'
    # anno_dir = img_dir
    # dst_dir = '/home/cuongnd/PycharmProjects/aicr/viText/viText/viData/viReceipts/split/20210323'
    # split_receipt_from_raw_img(img_dir=img_dir, anno_dir=anno_dir, dst_dir=dst_dir)

    img_dir = '/home/cuongnd/PycharmProjects/aicr/viText/viText/viData/viReceipts/split/images/difficult'
    anno_dir='/home/cuongnd/PycharmProjects/aicr/viText/viText/viData/viReceipts/split/anno'
    dst_dir = '/home/cuongnd/PycharmProjects/aicr/viText/viText/viData/viReceipts/split/images/new_di'
    rename_viReceipts(img_dir=img_dir,
                      anno_dir=anno_dir,
                      dst_dir=dst_dir)
