import json as js
import pycocotools.mask as maskUtils
import cv2
import numpy as np
from skimage.measure import find_contours
import argparse
import sys, time
from tqdm import *

def arg_parse():
    parser = argparse.ArgumentParser(description='Start Training')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--origin_result', default='../result/result.pkl.json', type=str)
    parser.add_argument('--final_result', default='../result/result_final.json', type=str)
    args = parser
    return args

global args
args = arg_parse().parse_args()
origin_path = args.origin_result
final_path = args.final_result
gt_path = '../data/art/train_labels.json'
thresh = args.thresh
conf_score = args.score

# 初始化
res = {}
start = 4000
end = 5603
EPS = 0.00001
for i in range(start, end):
    res['res_'+str(i)] = []

def point_convert(rle):
    decoded_mask = maskUtils.decode(rle)
#     print(decoded_mask.shape)
    mask = np.zeros((decoded_mask.shape[0]+2, decoded_mask.shape[1]+2), dtype = np.uint8)
    mask[1:-1, 1:-1] = decoded_mask
    contours = find_contours(mask, 0.5)
#     print(np.fliplr(contours[0]).tolist())
    area = float(maskUtils.area(rle))
#     print("area", area)
    if len(contours) == 0:
        return [[]], area
    points = np.fliplr(contours[0]).tolist()
    return points, area

def get_mask(box, shape):
    # 根据box（json文件解码的轮廓点），绘制mask
    tmp_mask = np.zeros(shape, dtype=np.uint8)
    tmp = np.array(box, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(tmp_mask, [tmp], (255))
    return tmp_mask, cv2.countNonZero(tmp_mask)

def MMI(area1, area2, intersect):
    if area1 == 0 or area2 == 0:
        area1 += EPS
        area2 += EPS
    return max(float(intersect)/area1, float(intersect)/area2)


def mask_nms(res_images, shape, conf=conf_score, thre=thresh):
    bbox_info = []
    # area = []
    scores = []
    for res in res_images:
        if res['confidence'] > conf:
            bbox_info.append(res['points'])
            # area.append(res['area'])
            scores.append(res['confidence'])
    # print("bbox_info:", len(bbox_info))
    tmp_final = []
    scores_sorted = np.array(scores).argsort()[::-1]  # 翻转，从大到小排序,返回索引
    # print(scores_sorted)
    num = len(bbox_info)
    # print(num)
    suppressed = np.zeros((num), dtype=np.int)

    for i in range(num):
        idx = scores_sorted[i]
        if suppressed[idx] == 1:
            print("i", i)
            continue
        tmp_final.append(idx)
        mask1_box, num1_nozero = get_mask(bbox_info[idx], shape)
        # print(mask1_box, num1_nozero)
        for j in range(i, num):
            idx_j = scores_sorted[j]
            if suppressed[idx_j] == 1:
                continue
            mask2_box, num2_nozero = get_mask(bbox_info[idx_j], shape)
            merge_mask = cv2.bitwise_and(mask1_box, mask2_box)
            intersect_area = cv2.countNonZero(merge_mask)
            mmi = MMI(num1_nozero, num2_nozero, intersect_area)
            # print("num1_nozero:{},num2_nozero:{},inte:{},mmi:{}".format(num1_nozero, num2_nozero, intersect_area, mmi))
            if mmi >= thre:
                suppressed[idx_j] = 1
    final_sets = []
    # print(len(tmp_final))
    for i in tmp_final:
        final_sets.append({
            'points': bbox_info[i],
            'confidence': scores[i]
        })
    # print("final_sets", len(final_sets))
    return final_sets

if __name__ == '__main__':
    with open(origin_path) as f:
        label_origin = js.load(f)
    l = len(label_origin)
    for index, rle in zip(tqdm(range(l)), label_origin):
        # if rle['image_id'] != 4000:
        #     continue
        points, area = point_convert(rle['segmentation'])
        if len(points[0]) == 0:
            print("point=0")
        image_name = 'res_' + str(rle['image_id'])
        res[image_name].append({
            'points': points,
            'confidence': rle['score'],
            'area': area,
            'size': rle['segmentation']['size']
        })
        # print("index:", index)
    for i in tqdm(range(start, end)):
        img_name = 'res_' + str(i)
        # print(image_name)
        if len(res[img_name]) == 0:
            continue
        res[img_name] = mask_nms(np.array(res[img_name]), res[img_name][0]['size'])
        # break
    with open(final_path, 'w') as f:
        js.dump(res, f)
        print(1)

