import os

from PIL import Image
from tqdm import tqdm
import shutil

from utils import get_classes
from utils_map import get_coco_map, get_map
from frcnn_predict import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    mode = "dir_predict"
    
    dir_origin_path = "My_Datasets/Detection/DeepPCB_VOC"
    
    dir_save_path   = "My_PCB_det/result1/input"
    dr_dir = os.path.join(dir_save_path, "detection-results")
    gt_dir = os.path.join(dir_save_path, "ground-truth")
    imgs_dir = os.path.join(dir_save_path, "images-optional")
    
    if not os.path.exists(dr_dir):
        os.makedirs(dr_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
    class_names = get_classes(dir_origin_path + '/cls_classes.txt')
    
    images = [i.strip() for i in open(dir_origin_path+'/test.txt', 'r').readlines()]
    image_ids = [i.split(' ')[0].split('/')[-1][:-4] for i in images]
    
    for image_id in image_ids:
        
        if dir_origin_path.split('/')[-1] in ["PCB_瑕疵初赛样例集_VOC"]:
            shutil.copy(os.path.join(dir_origin_path, "Annotations", image_id+'.txt'), os.path.join(gt_dir, image_id+'.txt'))
            shutil.copy(os.path.join(dir_origin_path, "ImageSets", image_id+'.bmp'), os.path.join(imgs_dir, image_id+'.bmp'))
        else:
            shutil.copy(os.path.join(dir_origin_path, "Annotations", image_id+'.txt'), os.path.join(gt_dir, image_id+'.txt'))
            shutil.copy(os.path.join(dir_origin_path, "ImageSets", image_id+'.jpg'), os.path.join(imgs_dir, image_id+'.jpg'))
        
        image_path  = os.path.join(imgs_dir, image_id+'.bmp')
        print(image_path)
        image       = Image.open(image_path)
        map_txt     = frcnn.get_map_txt(image_id, image, class_names, dir_save_path)
    