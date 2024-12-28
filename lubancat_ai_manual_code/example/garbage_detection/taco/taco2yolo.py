import os
from pycocotools.coco import COCO
import numpy as np
import tqdm
import argparse
import shutil

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str,default='./data/annotations.json', help='dataset annotations')
    parser.add_argument('--save_path', type=str, default='./taco')
    parser.add_argument('--subset', type=str, default='train', required=False, help='which subset(train, val, test)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    annotation_path = args.annotation_path
    yolo_image_path = args.save_path + '/images/' + args.subset
    yolo_label_path = args.save_path +'/labels/' + args.subset

    os.makedirs(yolo_image_path, exist_ok=True)
    os.makedirs(yolo_label_path, exist_ok=True)

    data_source = COCO(annotation_file=annotation_path)

    # 类别ID
    catIds = data_source.getCatIds()
    # 获取类别名称
    categories = data_source.loadCats(catIds)
    categories.sort(key=lambda x: x['id'])

    # 保存类别
    class_path = args.save_path + '/classes.txt'
    with open(class_path, "w") as file:
        for item in categories:
            file.write(f"{item['id']}: {item['name']}\n")

    #遍历每张图片
    img_ids = data_source.getImgIds()
    for index, img_id in tqdm.tqdm(enumerate(img_ids)):
        img_info = data_source.loadImgs(img_id)[0]
        file_name = img_info['file_name'].replace('/', '_')
        save_name = file_name.split('.')[0]

        height = img_info['height']
        width = img_info['width']

        save_label_path = yolo_label_path + '/' + save_name + '.txt'
        with open(save_label_path, mode='w') as fp:
            annotation_id = data_source.getAnnIds(img_id)
            if len(annotation_id) == 0:
                fp.write('')
                shutil.copy('data/{}'.format(img_info['file_name']), os.path.join(yolo_image_path, file_name))
                continue

            annotations = data_source.loadAnns(annotation_id)

            for annotation in annotations:
                category_id = annotation["category_id"]
                seg_labels = []
                for segmentation in annotation["segmentation"]:
                    points = np.array(segmentation).reshape((int(len(segmentation) / 2), 2))
                    for point in points:
                        x = point[0] / width
                        y = point[1] / height
                        seg_labels.append(x)
                        seg_labels.append(y)
                fp.write(str(category_id) + " " + " ".join([str(a) for a in seg_labels]) + "\n")

        shutil.copy('data/{}'.format(img_info['file_name']), os.path.join(yolo_image_path, file_name))

