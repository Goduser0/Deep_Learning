# -*- coding=utf-8 -*-
# !/usr/bin/python

import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import mmcv
from tqdm import tqdm
from glob import glob
import argparse
#from mmdet.apis import project_dir
START_BOUNDING_BOX_ID = 1
#
START_IMAGE_ID = 1
USE_NAME_AS_IMAGE_ID = True
WITHOUT_IMAGE_INFO = True

# construct box  for negative image
NEGATIVE_BOX = True

# PRE_DEFINE_CATEGORIES = {"1_chongkong":1,"2_hanfeng":2,"3_yueyawan":3,"4_shuiban":4,"5_youban":5,
#                         "6_siban":6,"7_yiwu":7,"8_yahen":8,"9_zhehen":9,"10_yaozhe":10}

PRE_DEFINE_CATEGORIES = {"crazing":1,"inclusion":2,"patches":3,"pitted_surface":4,"rolled-in_scale":5,"scratches":6}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def convert(xml_list, xml_dir, image_dir, json_file,use_legacy_coordinate=True):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    # 标注基本结构
    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    # image_id = START_IMAGE_ID - 1
    for line in tqdm(list_fp):
        line = line.strip()
        # print("buddy~ Processing {}".format(line))
        # 解析XML
        print(line)
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'frame')
        # 取出图片名字
        filename = line.split('.')[0]
        ## The filename must be a number
        # if USE_NAME_AS_IMAGE_ID:
        #     image_id = get_filename_as_int(filename)  # 图片ID
        # else:
        #     image_id += 1
        # image_id = get_filename_as_int(filename)
        image_id = list_fp.index(line)+1
        # print(image_id)    
        try:
            filename += '.jpg'
            img_path = os.path.join(image_dir, filename)
            img = mmcv.imread(img_path)
        except FileNotFoundError:
            filename = line.split('.')[0] + '.JPG'
            img_path = os.path.join(image_dir, filename)
            img = mmcv.imread(img_path)
        height, width, _ = img.shape
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id': image_id}

        # # 处理每个标注的检测框
        if len(get(root, 'object')) == 0 and NEGATIVE_BOX:
        #     # negative image
            image['negative'] = 1
        #     #assert filename[:2] in ('ss', 'fl', 'gx')
        #     xmin, ymin, xmax, ymax = 0, 0, width, height
        #     annotation = dict()
        #     annotation['area'] = width * height
        #     annotation['iscrowd'] = 0
        #     annotation['image_id'] = image_id
        #     annotation['bbox'] = [xmin, ymin, width, height]
        #     #category_id 设置图片所属类
        #     ## 前视————1；侧视————flv-2
        #     annotation['category_id'] = 2
        #     annotation['id'] = bnd_id
        #     annotation['ignore'] = 0  
        #     # flag
        #     annotation['negative'] = 1
        #     # 设置分割数据，点的顺序为逆时针方向
        #     annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]

        #     json_dict['annotations'].append(annotation)
        #     bnd_id = bnd_id + 1
        #     # continue
        else:
            if NEGATIVE_BOX:
                image['negative'] = 0

        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            # 取出检测框类别名称
            
            category = get_and_check(obj, 'name', 1).text
            
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            if xmax - xmin <= 2 or ymax - ymin <= 2:
                continue
            assert (xmax > xmin)
            assert (ymax > ymin) 
            if use_legacy_coordinate:
                o_width = abs(xmax - xmin)+1
                o_height = abs(ymax - ymin)+1
            else:
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)                
            annotation = dict()
            annotation['area'] = o_width * o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            if category == "10_yaozhed":
                annotation['category_id'] = PRE_DEFINE_CATEGORIES["10_yaozhe"]
            else:
                annotation['category_id'] = PRE_DEFINE_CATEGORIES[category]
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # flag
            if NEGATIVE_BOX:
                annotation['negative'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    print(json_dict['categories'])
    print('end image id: {} total images: {}'.format(image_id, len(json_dict['images'])))
    print('eng bnx id: {} total instances: {}'.format(bnd_id, len(json_dict['annotations'])))
    if NEGATIVE_BOX:
        print('negative images : ', sum([a['negative'] for a in json_dict['images']]))
        # for a in json_dict['images']:
        #     if a['negative']:
        #         print(a['file_name'])
    print("statistic the distribution of instance :")
    for name, id in PRE_DEFINE_CATEGORIES.items():
        ins_sum = 0
        for ann in json_dict['annotations']:
            if ann['category_id'] == id:
                ins_sum += 1
        print('class {}: {} instances'.format(name, ins_sum))
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir",default='',type = str,help="path for all xml file")
    parser.add_argument("--img_dir",default='',type = str,help="path for all img file")
    parser.add_argument("--save_dir",default='',type = str,help="path for saving all coco file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    xml_dir = args.xml_dir
    img_dir = args.img_dir
    json_dir = args.save_dir
    xml_list = os.listdir(xml_dir)

    # all_xml_list = dict(train=[],val=[],test=[])

    # np.random.shuffle(xml_list)

    # split_point = int(len(xml_list)/5)
    # print(split_point)

    # all_xml_list['test'].extend(xml_list[:split_point])
    # all_xml_list['val'] = all_xml_list['test']
    # all_xml_list['train'].extend(xml_list[split_point:len(xml_list)])

    xml_list = sorted(xml_list)

    # for ind in ('train', 'val',"test"):
    #     json_file = os.path.join(json_dir,ind + '.json')
    #     if not os.path.exists(json_dir):
    #         os.mkdir(json_dir)
    #     convert(all_xml_list[ind], xml_dir, img_dir, json_file)

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    json_file = os.path.join(json_dir, 'instances_all.json')
    convert(xml_list, xml_dir, img_dir, json_file)


