import os
import shutil
import cv2
from tqdm import tqdm
from PIL import Image

from pandas import read_csv
from shutil import move, copy
from random import sample
from tqdm import tqdm

import zipfile

# 强迫症表示必须按照文件夹排序顺序定义函数
# 后面加函数请对应好位置
# 先下载原始数据集，放到Dataset下，修改root/source/target路径后再执行函数

#######################################
# utils
#######################################

def jpeg2jpg(path_in, path_out):
    img = Image.open(path_in)
    img.save(path_out, "JPEG", optimize=True, progressive=True)
    
def bmp2jpg(path_in, path_out):
    img = Image.open(path_in)
    img.save(path_out)
    
def jpgResize(path_in, path_out, img_size):
    img = cv2.imread(path_in, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC) # bicubic
    cv2.imwrite(path_out, img_resize)    

def randomSplit(source, target, ratio=0.5, mode="move"):
    # move images from source to target, split train and test
    os.makedirs(target, exist_ok=True)

    img_cls = os.listdir(source)
    for _cls in img_cls:
        targrt_cls = os.path.join(target, _cls)
        os.makedirs(targrt_cls, exist_ok=True)
        source_cls = os.path.join(source, _cls)

        move_num = int(len(os.listdir(source_cls)) * ratio)
        move_img = sample(os.listdir(source_cls), move_num)
        
        if mode == "move":
            for img in tqdm(move_img):
                move(source_cls + '/' + img, targrt_cls + '/' + img)
        if mode == "copy":
            for img in tqdm(move_img):
                copy(source_cls + '/' + img, targrt_cls + '/' + img)

#######################################
# build datasets functions
#######################################

def build_bridge_crack_image_dataset():
    train_source = "/home/user/duzongwei/Dataset/Bridge_Crack_Image/DBCC_Training_Data_Set/train"
    val_source   = "/home/user/duzongwei/Dataset/Bridge_Crack_Image/DBCC_Training_Data_Set/val"
    train_target = "/home/user/duzongwei/Dataset/Classification/Bridge_Crack_Image/train"
    val_target   = "/home/user/duzongwei/Dataset/Classification/Bridge_Crack_Image/val"
    
    labels = ["cr", "no"]
    for label in labels:
        os.makedirs(train_target + "/" + label, exist_ok=True)
        os.makedirs(val_target + "/" + label, exist_ok=True)
    
    # move train and 2jpg
    for img in tqdm(os.listdir(train_source)):
        path_in = os.path.join(train_source, img)
        class_ = img.split('.')[0][:2]
        name_  = img.split('.')[0]
        if class_ == "cr":
            path_out = train_target + '/cr/' + name_ + ".jpg"
            jpeg2jpg(path_in, path_out)
        if class_ == "no":
            path_out = train_target + '/no/' + name_ + ".jpg"
            jpeg2jpg(path_in, path_out)
    # move val and 2jpg
    for img in tqdm(os.listdir(val_source)):
        path_in = os.path.join(val_source, img)
        class_ = img.split('.')[0][:2]
        name_  = img.split('.')[0]
        if class_ == "cr":
            path_out = val_target + '/cr/' + name_ + ".jpg"
            jpeg2jpg(path_in, path_out)
        if class_ == "no":
            path_out = val_target + '/no/' + name_ + ".jpg"
            jpeg2jpg(path_in, path_out)
    # train: cr/no  6000/44000
    # val:   cr/no  1000/4000
# build_bridge_crack_image_dataset()

def build_elpv_dataset():
    root = "/home/user/duzongwei/Dataset/elpv-dataset"
    source = "/home/user/duzongwei/Dataset/Classification/elpv/train"
    os.makedirs(source, exist_ok=True)
    
    labels_path = root + "/labels.csv"
    labels = ["mono", "poly"]
    for label in labels:
        os.makedirs(source + "/" + label, exist_ok=True)
    
    infomations = read_csv(labels_path, names=["0"])
    for i in tqdm(range(len(infomations))):
        info = infomations.iloc[i, 0].split()
        img_path = os.path.join(root, info[0])
        target = info[-1]
        if target == labels[0]:
            # copy to mono
            copy(img_path, source + '/mono')
        if target == labels[1]:
            # copy to poly
            copy(img_path, source + '/poly')
    
    # mono/poly = 1074/1550 -> 537/775
    # split train and test as 1:1
    randomSplit(source, '/'.join(source.split('/')[:-1]) + '/val')
# build_elpv_dataset()

def build_kth_gray_dataset():
    source = "/home/user/duzongwei/Dataset/Classification/KTH-TIPS/train"
    target = "/home/user/duzongwei/Dataset/Classification/KTH-TIPS/val"
    
    # copy /home/user/duzongwei/Dataset/KTH-TIPS/kth_tips_grey_200x200 to 
    # /home/user/duzongwei/Dataset/Classification/KTH-TIPS/train
    
    # split train/test 41/40
    randomSplit(source, target)
# build_kth_gray_dataset()

def build_kylberg_textture_dataset():
    # First you need to unzip all classes at terminal
    root = "/home/user/duzongwei/Dataset/Kylberg-Texture-Dataset"
    source = "/home/user/duzongwei/Dataset/Classification/Kylberg_Texture_Dataset/train"
    target = "/home/user/duzongwei/Dataset/Classification/Kylberg_Texture_Dataset/val"
    # unzip to source
    for file in tqdm(os.listdir(root)):
        if file.split('.')[1] == "zip":
            zip_file = zipfile.ZipFile(root + '/' + file)
            unzip_file = source + '/' + file.split('.')[0]
            os.makedirs(unzip_file, exist_ok=True)
            for names in zip_file.namelist():
                zip_file.extract(names, unzip_file)
            zip_file.close()
    # split train and tesr as 1:1
    # number of classes==28, number of unique samples==160, split train/test==80/80
    randomSplit(source, target)
# build_kylberg_textture_dataset()

def build_magnetic_tile_defect_dataset():
    # just nedd jpg
    root = "/home/user/duzongwei/Dataset/Magnetic-Tile-Defect"
    source = "/home/user/duzongwei/Dataset/Classification/Magnetic_Tile_Defect/train"
    targrt = "/home/user/duzongwei/Dataset/Classification/Magnetic_Tile_Defect/val"
    # move to train
    for cls_ in os.listdir(root):
        if os.path.isdir(root + '/' + cls_):
            dst_path = source + '/' + cls_
            # build class dir and move each class
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            imgs_path = root + '/' + cls_ + "/Imgs"
            for img in tqdm(os.listdir(imgs_path)):
                if img.split('.')[1] == "jpg":
                    move(src=imgs_path + '/' + img, dst=dst_path + '/' + img)
    # split train and test 58/16/476/29/52/43
    randomSplit(source, targrt)
# build_magnetic_tile_defect_dataset()

def build_gc10_det_dataset():
    source = "/home/user/duzongwei/Dataset/Classification/GC10-DET/train"
    target = "/home/user/duzongwei/Dataset/Classification/GC10-DET/val"
    
    # copy imgs
    # split train/test
    randomSplit(source, target)
# build_gc10_det_dataset()

def build_neu_dataset():
    NEU_CLS = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-CLS"
    NEU_prime = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-prime"
    NEU_prime128 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-prime128"
    NEU_300 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-300"
    NEU_150 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-150"
    NEU_100 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-100"
    NEU_50  = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-50"
    NEU_30  = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-30"
    NEU_10  = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-10"
    NEU_50_r64 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-50-r64"
    NEU_30_r64 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-30-r64"
    NEU_10_r64 = "/home/user/duzongwei/Dataset/Classification/NEU/NEU-10-r64"
    
    os.makedirs(NEU_prime, exist_ok=True)
    os.makedirs(NEU_prime128, exist_ok=True)
    os.makedirs(NEU_300, exist_ok=True)
    os.makedirs(NEU_150, exist_ok=True)
    os.makedirs(NEU_100, exist_ok=True)
    os.makedirs(NEU_50, exist_ok=True)
    os.makedirs(NEU_30, exist_ok=True)
    os.makedirs(NEU_10, exist_ok=True)
    os.makedirs(NEU_50_r64, exist_ok=True)
    os.makedirs(NEU_30_r64, exist_ok=True)
    os.makedirs(NEU_10_r64, exist_ok=True)
    
    
    # build NEU-prime, jpg, 200x200
    # for bmp in os.listdir(NEU_CLS):
    #     path_in = os.path.join(NEU_CLS, bmp)
    #     path_out = os.path.join(NEU_prime, bmp.split('.')[0] + ".jpg")
    #     bmp2jpg(path_in, path_out)
        
    # build NEU-prime128, jpg, 128x128
    # for jpg in os.listdir(NEU_prime):
    #     path_in = os.path.join(NEU_prime, jpg)
    #     path_out = os.path.join(NEU_prime128, jpg)
    #     jpgResize(path_in, path_out, img_size=(128, 128))
    
    # build NEU-300, split class dir
    # labels = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
    # for i in labels:
    #     os.makedirs(os.path.join(NEU_300, i), exist_ok=True)
    # for jpg in os.listdir(NEU_prime128):
    #     shutil.copy(NEU_prime128 + '/' + jpg, NEU_300 + '/' + jpg[0:2])
    
    # build NEU-150
    # if not os.path.exists(NEU_150 + "/train"):
    #     shutil.copytree(NEU_300, NEU_150 + "/train")
    # randomSplit(NEU_150 + "/train", NEU_150 + "/test")
    
    # build NEU-100
    # if not os.path.exists(NEU_100 + "/test"):
    #     shutil.copytree(NEU_150 + "/test", NEU_100 + "/test")
    # randomSplit(NEU_150 + "/train", NEU_100 + "/train", mode="copy", ratio=100/150)
    
    # build NEU-50
    # if not os.path.exists(NEU_50 + "/test"):
    #     shutil.copytree(NEU_150 + "/test", NEU_50 + "/test")
    # randomSplit(NEU_150 + "/train", NEU_50 + "/train", mode="copy", ratio=50/150)
    
    # build NEU-30
    # if not os.path.exists(NEU_30 + "/test"):
    #     shutil.copytree(NEU_150 + "/test", NEU_30 + "/test")
    # randomSplit(NEU_150 + "/train", NEU_30 + "/train", mode="copy", ratio=30/150)

    # build NEU-10
    # if not os.path.exists(NEU_10 + "/test"):
    #     shutil.copytree(NEU_150 + "/test", NEU_10 + "/test")
    # randomSplit(NEU_150 + "/train", NEU_10 + "/train", mode="copy", ratio=10/150)
    
    # build NEU-30-r64, resolution 64x64
    for i in os.listdir(NEU_30):
        # train / test
        os.makedirs(NEU_30_r64 + '/' + i, exist_ok=True)
        for c in os.listdir(NEU_30 + '/' + i):
            # class
            save_path = NEU_30_r64 + '/' + i + '/' + c
            os.makedirs(save_path, exist_ok=True)
            for jpg in os.listdir(NEU_30 + '/' + i + '/' + c):
                path_in = os.path.join(NEU_30 + '/' + i + '/' + c, jpg)
                path_out = os.path.join(save_path, jpg)
                jpgResize(path_in, path_out, img_size=(64, 64))
    
    # build NEU-10-r64, resolution 64x64
    for i in os.listdir(NEU_10):
        # train / test
        os.makedirs(NEU_30_r64 + '/' + i, exist_ok=True)
        for c in os.listdir(NEU_10 + '/' + i):
            # class
            save_path = NEU_10_r64 + '/' + i + '/' + c
            os.makedirs(save_path, exist_ok=True)
            for jpg in os.listdir(NEU_10 + '/' + i + '/' + c):
                path_in = os.path.join(NEU_10 + '/' + i + '/' + c, jpg)
                path_out = os.path.join(save_path, jpg)
                jpgResize(path_in, path_out, img_size=(64, 64))
                
build_neu_dataset()
