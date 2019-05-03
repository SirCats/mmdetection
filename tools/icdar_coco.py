import json as js
import cv2
import pycocotools.mask as maskUtils
from tqdm import *


LICENSES = {
    'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
    'id': 1,
    'name': 'Attribution-NonCommercial-ShareAlike License'
}
INFO = {
    'description': 'train Dataset',
    'url': 'http://rrc.cvc.uab.es',
    'version': '1.0',
    'year': 2019,
    'contributor': 'icdar',
    'data_created': '2019/03/01'
}
CATEGORIES = {
    'id': 1,
    'name': 'text',
    'supercategory': 'text'
}
data = {
    "info": INFO,
    "images": [],
    "licenses": LICENSES,
    "annotations": [],
    "categories": [CATEGORIES]
}
from_path = '../data/art/train_labels.json'
to_path = '../data/art/train_label_coco.json'
def to_list(points):
    point_list = []
    for p in points:
        x = float(p[0])
        y = float(p[1])
        point_list.append(x)
        point_list.append(y)
    return point_list


if __name__ == '__main__':
    with open(from_path) as f:
        label = js.load(f)

    an_id = 0
    l = len(label)
    print('len', l)
    for index, img in zip(tqdm(range(l)), label):
        # print("index")
        image = cv2.imread('../data/art/train/' + img + '.jpg')

        data['images'].append({
            'license': 1,
            'file_name': img + '.jpg',
            'width': image.shape[1],
            'height': image.shape[0],
            'id': int(img[3:])
        })

        for an_index, anno in enumerate(label[img]):
            if anno['transcription'] == '###':
                continue
            if anno['illegibility'] == True:
                continue
            seg = [to_list(anno['points'])]
            rle = maskUtils.frPyObjects(seg, float(image.shape[0]), float(image.shape[1]))
            area = float(maskUtils.area(rle[0]))
            #         print(rle)
            bbox = maskUtils.toBbox(rle[0])
            #         print(maskUtils.toBbox(rle)[0])
            data['annotations'].append({
                'id': an_id,
                'image_id': int(img[3:]),
                'category_id': 1,
                'segmentation': seg,
                'area': area,
                'bbox': bbox.tolist(),
                'iscrowd': 0
            })
            an_id = an_id + 1
            # break
        # break

    with open(to_path, 'w') as f:
        js.dump(data, f)
        print("completed!")

