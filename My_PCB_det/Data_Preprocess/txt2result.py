import os
import shutil

dataset_name = "DeepPCB_VOC"
assert dataset_name in ["DeepPCB_VOC", "PCB_dataset_VOC", "PCB_瑕疵初赛样例集_VOC"]

root = "My_Datasets/Detection/DeepPCB_VOC/PCBData"
anno_path = "My_Datasets/Detection/DeepPCB_VOC/Annotations"
imgs_path = "My_Datasets/Detection/DeepPCB_VOC/ImageSets"

for group in os.listdir(root):
    for imgs in os.listdir(os.path.join(root, group, group[-5:])):
        if imgs.split('_')[1][:4] == 'test':
            shutil.copy(os.path.join(root, group, group[-5:], imgs), os.path.join(imgs_path, imgs[:8]+'.jpg'))
    for txts in os.listdir(os.path.join(root, group, group[-5:]+'_not')):
        shutil.copy(os.path.join(root, group, group[-5:]+'_not',txts), os.path.join(anno_path, txts))