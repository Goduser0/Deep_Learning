import os
import sys
import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

sys.path.append('My_PCB_det/Utils')
from utils import get_classes

annotation_mode = 0

classes_path = "My_Datasets/Detection/PCB_dataset_VOC/cls_classes.txt"

# (训练集+验证集):测试集 = 9:1
train_val_percent = 0.9
# 训练集:测试集 = 9:1
train_percent = 0.9

PCB_DataSet_path = "My_Datasets/Detection/PCB_dataset_VOC"
PCB_Data_Sets = ['trainval', 'test']

classes, _ = get_classes(classes_path)
print(classes)

photo_nums = np.zeros(len(PCB_Data_Sets))
nums = np.zeros(len(classes))

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(PCB_DataSet_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格")
    
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(PCB_DataSet_path, "Annotations")
        saveBasepath = os.path.join(PCB_DataSet_path, "ImageSets")
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num*train_val_percent)
        tr = int(tv*train_percent)
        
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)
        
        print(f"train and val size:{tv}; train size:{tr}")
        
        ftrainval = open(os.path.join(saveBasepath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasepath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasepath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasepath, 'val.txt'), 'w')

        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name) 

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets Done.")
        
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate PCB_train.txt and PCB_val.txt for train.")
        type_index = 0
        
        for image_set in PCB_Data_Sets:
            image_ids = open(os.path.join(PCB_DataSet_path, 'ImageSets/%s.txt'%(image_set)), encoding='utf-8').read().strip().split()
            list_file = open(os.path.join(PCB_DataSet_path,'%s.txt'%( image_set)), 'w', encoding='utf-8')#保存训练集和测试集
            for image_id in image_ids:
                img_path = '%s/ImageSets/%s.jpg'%(os.path.abspath(PCB_DataSet_path),image_id)
                img = Image.open(img_path).convert('RGB')
                height, width = np.array(img).shape[:2]
                
                list_file.write(img_path)#在训练集和测试集中写入图片路径信息
                
                in_file = open(os.path.join(PCB_DataSet_path, 'Annotations/%s.xml'%(image_id)), encoding='utf-8')
                tree = ET.parse(in_file)
                root = tree.getroot()
                
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    content = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
                    list_file.write(" " + ",".join([str(a) for a in content]) + ',' + str(cls_id))
                    
                    nums[classes.index(cls)] = nums[classes.index(cls)] + 1
                #
                list_file.write('\n')
            
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate PCB_train.txt and PCB_val.txt for train done.")

        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500,属于较小的数据量,请注意设置较大的训练世代(Epoch)以满足足够的梯度下降次数(Step)。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标,请注意修改classes_path对应自己的数据集")