import os
import shutil

dataset_name = ["DeepPCB_VOC", "PCB_dataset_VOC", "PCB_瑕疵初赛样例集_VOC"]
assert all([(i in ["DeepPCB_VOC", "PCB_dataset_VOC", "PCB_瑕疵初赛样例集_VOC"]) for i in dataset_name])
tar_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

if "DeepPCB_VOC" in dataset_name:
    #1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole
    trans_ID = [1, 2, 0, 3, 4, -1]
    root = "My_Datasets/Detection/DeepPCB_VOC/PCBData"
    anno_path = "My_Datasets/Detection/DeepPCB_VOC/Annotations"
    imgs_path = "My_Datasets/Detection/DeepPCB_VOC/ImageSets"
    if os.path.exists(anno_path):
        shutil.rmtree(anno_path)
    if os.path.exists(imgs_path):
        shutil.rmtree(imgs_path)
    os.makedirs(anno_path)
    os.makedirs(imgs_path)

    for group in os.listdir(root):
        for txt in os.listdir(os.path.join(root, group, group[-5:]+'_not')):
            with open(os.path.join(root, group, group[-5:]+'_not', txt), 'r', encoding='utf-8') as old_f:
                contents = [i.strip() for i in old_f.readlines()]
            old_f.close()
            with open(os.path.join(anno_path, txt), 'w', encoding='utf-8') as new_f:
                add_img_flag = -1
                for obj in contents:
                    if int(trans_ID[int(obj[-1])-1]) != -1:
                        new_f.write(obj[:-1]+str(trans_ID[int(obj[-1])-1])+'\n')
                        add_img_flag = 1
            new_f.close()

            if add_img_flag == 1:
                shutil.copy(os.path.join(root, group, group[-5:], txt.split('.')[0]+'_test.jpg'), os.path.join(imgs_path, txt.split('.')[0]+'.jpg'))
        
elif "PCB_dataset_VOC" in dataset_name:
    src_anno_path = "My_Datasets/Detection/PCB_dataset/Annotations"
    src_imgs_path = "My_Datasets/Detection/PCB_dataset/images"
    
    tar_anno_path = "My_Datasets/Detection/PCB_dataset_VOC/Annotations"
    tar_imgs_path = "My_Datasets/Detection/PCB_dataset_VOC/ImageSets"
    os.makedirs(tar_anno_path)
    os.makedirs(tar_imgs_path)

    for group in os.listdir(src_anno_path):
        if group != "Missing_hole":
            for imgs in os.listdir(os.path.join(src_imgs_path, group)):
                shutil.copy(os.path.join(src_imgs_path, group, imgs), os.path.join(tar_imgs_path, imgs))
                shutil.copy(os.path.join(src_anno_path, group, imgs.split('.')[0]+'.xml'), os.path.join(tar_anno_path, imgs.split('.')[0]+'.xml'))
            
elif "PCB_瑕疵初赛样例集_VOC" in dataset_name:
    src_path = "My_Datasets/Detection/PCB_瑕疵初赛样例集"
    tar_anno_path = "My_Datasets/Detection/PCB_瑕疵初赛样例集_VOC/Annotations"
    tar_imgs_path = "My_Datasets/Detection/PCB_瑕疵初赛样例集_VOC/ImageSets"
    os.makedirs(tar_anno_path)
    os.makedirs(tar_imgs_path)

    for group in os.listdir(src_path):
        if group[-4:] == "_Img":
            for imgs in os.listdir(os.path.join(src_path, group)):
                shutil.copy(os.path.join(src_path, group, imgs), os.path.join(tar_imgs_path, imgs))
                shutil.copy(os.path.join(src_path, group[:-3]+'txt', imgs.split('.')[0]+'.txt'), os.path.join(tar_anno_path, imgs.split('.')[0]+'.txt'))