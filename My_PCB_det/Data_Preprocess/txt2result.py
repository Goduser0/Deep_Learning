import os
import shutil
import sys

sys.path.append('My_PCB_det/Utils')
from utils import get_classes

dataset_name = "PCB_瑕疵初赛样例集_VOC"
assert dataset_name in ["DeepPCB_VOC", "PCB_dataset_VOC", "PCB_瑕疵初赛样例集_VOC"]

src_path = "My_Datasets/Detection/%s"%(dataset_name)

if os.path.exists("My_PCB_det/Results/%s"%(dataset_name)):
    shutil.rmtree("My_PCB_det/Results/%s"%(dataset_name))
tar_anno_path = "My_PCB_det/Results/%s/input/ground-truth"%(dataset_name)
tar_imgs_path = "My_PCB_det/Results/%s/input/images-optional"%(dataset_name)
os.makedirs(tar_imgs_path)
os.makedirs(tar_anno_path)

classes, _ = get_classes(os.path.join(src_path, "cls_classes.txt"))
shutil.copy(os.path.join(src_path, "cls_classes.txt"), "My_PCB_det/Results/%s/cls_classes.txt"%(dataset_name))
print(classes)


with open(os.path.join(src_path, "test.txt"), encoding='utf-8') as f:
    test_lines = [i.strip() for i in f.readlines()]

for test_line in test_lines:
    contents = test_line.split(' ')
    img_path = contents[0]
    img_id = img_path.split('/')[-1].split('.')[0]
    objects = contents[1:]
    
    if dataset_name in ["PCB_瑕疵初赛样例集_VOC"]:
        shutil.copy(img_path, os.path.join(tar_imgs_path, img_id+'.bmp'))
    elif dataset_name in ["DeepPCB_VOC", "PCB_dataset_VOC"]:
        shutil.copy(img_path, os.path.join(tar_imgs_path, img_id+'.jpg'))
    
    txt_file = open(os.path.join(tar_anno_path, img_id+'.txt'), 'w', encoding='utf-8')
    for obj in objects:
        obj = obj.split(',')
        txt_file.write(str(int(obj[-1]))+' '+" ".join([str(i) for i in obj[:-1]]) + '\n')
    txt_file.close()
        