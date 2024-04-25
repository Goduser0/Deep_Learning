import os
import cv2 as cv
# DeepPCB
root_dir = "My_Datasets/Detection/DeepPCB/PCBData"
save_dir = "My_Datasets/Classification/DeepPCB-Crop"

filelist = os.listdir(root_dir)
grouplist = []
for file in filelist:
    if file[:5] == "group":
        grouplist.append(file)

for group in grouplist:
    group_dir = os.path.join(root_dir, group)
    imgs_dir = os.path.join(group_dir, group[5:])
    anno_dir = os.path.join(group_dir, group[5:] + '_not')
    
    for anno_file in os.listdir(anno_dir):
        anno_file_dir = os.path.join(anno_dir, anno_file)
        img_ID = anno_file.split('.')[0]
        img_file_dir = os.path.join(imgs_dir, img_ID + "_test.jpg")
        
        img = cv.imread(img_file_dir, cv.IMREAD_COLOR)
        height = img.shape[0]
        weight = img.shape[1]
        
        # 从txt文件中读取缺陷坐标和类型
        locations = open(anno_file_dir, "r")
        locations = locations.readlines()
        count = 0
        for location in locations:
            xmin, ymin, xmax, ymax, label_index = [int(i) for i in location.split(" ")]
            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            
            label_list = ["Open_circuit", "Short", "Mouse_bite", "Spur", "Spurious_copper", "Pin_hole"]
            label = label_list[label_index-1]
            
            xmin = x_center-50 if x_center >= 50 else 0
            xmax = x_center+50 if x_center+50 <= weight else weight
            ymin = y_center-50 if y_center >= 50 else 0
            ymax = y_center+50 if y_center+50 <= height else height
            
            img_crop = img[ymin:ymax, xmin:xmax, :]
            
            save_path = os.path.join(save_dir, label)
            img_file_name = f"{label}_{img_ID}_{count}.jpg"
            count += 1
            save_path = os.path.join(save_path, img_file_name)
            
            cv.imwrite(save_path, img_crop)