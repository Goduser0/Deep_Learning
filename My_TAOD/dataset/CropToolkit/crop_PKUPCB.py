import os
import cv2 as cv
import xml.etree.ElementTree as ET
# PKUPCB
root_dir = "My_Datasets/Detection/PCB_dataset"
save_dir = "My_Datasets/Classification/PCB-Crop"

annotations_dir = os.path.join(root_dir, 'Annotations')
images_dir = os.path.join(root_dir, 'images')

label_list = os.listdir(annotations_dir)

for label in label_list:
    filefolder_dir = os.path.join(annotations_dir, label)
    filename_list = os.listdir(filefolder_dir)
    
    last_path = os.path.join(save_dir, label)
    os.makedirs(last_path, exist_ok=True)
    
    cout = 0
    
    for filename in filename_list:
        
        filepath = os.path.join(filefolder_dir, filename)
        # Start
        dom = ET.parse(filepath)
        root = dom.getroot()
        
        label = root.find('folder').text
        img_path = os.path.join(images_dir, label, root.find('filename').text)
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        
        height = img.shape[0]
        weight = img.shape[1]
        
        allobjects = root.findall("object")
        location_list = []
        # get objects
        for object in allobjects:
            if label.lower() == object.find("name").text.lower():
                bndbox = object.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                location_list.append([xmin, ymin, xmax, ymax])
        
        # Crop
        for i, location in enumerate(location_list):
            save_name = root.find('filename').text[:-4] +'_' + str(i) + '.jpg'
            save_path = os.path.join(last_path, save_name)
            
            # print(label, img_path, save_path, location)
            
            xmin = location[0]
            xmax = location[2]
            ymin = location[1]
            ymax = location[3]
            
            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            
            alpha_size = 25
            ymin = y_center-alpha_size if y_center >= alpha_size else 0
            ymax = y_center+alpha_size if y_center+alpha_size <= height else height
            
            xmin = x_center-alpha_size if x_center >= alpha_size else 0
            xmax = x_center+alpha_size if x_center+alpha_size <= weight else weight
                
            
            img_crop = img[ymin:ymax, xmin:xmax, :] # 2*alpha_size x 2*alpha_size 50x50
            # print(img_crop.shape)
            cv.imwrite(save_path, img_crop)
            cout += 1
           
    print(f"{label} | Total {cout} | Done!!!")
        
    
    

