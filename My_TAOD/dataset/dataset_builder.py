import os
import sys
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as T


###########################################################################################################
# 图像处理函数
###########################################################################################################
def bmp2ndarray(file_path):
    """将 BMP/PNG 文件转为 ndarray """
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array


###########################################################################################################
# DATASET CLASS: NEU_CLS
###########################################################################################################
class NEU_CLS(data.Dataset):
    """Dataset class for NEU_CLS"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/NEU-CLS'):
        self.root_dir = root_dir
        self.samples = self.load_samples()
        self.label_list = self.samples['Image_Class'].unique().tolist()
    
    def __len__(self):
        return len(self.samples)
     
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        # 开裂(Crazing)
        # 点蚀表面(Pitted Surface)
        # 内含物(Inclusion)
        # 轧制氧化皮(Rolled-in Scale)
        # 划痕(Scratches)
        # 斑块(Patches)
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label

    def load_samples(self):
        sample_list = []
        for filename in os.listdir(self.root_dir):
            filename_list = filename.split('.')
            if filename_list[1] == 'bmp':
                # image_path
                img_path = os.path.join(self.root_dir, filename)
                # image_class
                img_class = filename_list[0][:2]
                # image_label
                img_classes = ['Cr', 'PS', 'In', 'RS', 'Sc', 'Pa']
                img_label = img_classes.index(img_class)
                
                image_item = [img_label, img_class, img_path]
                sample_list.append(image_item)   
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])
    
    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/NEU_CLS.csv")

  
###########################################################################################################
# DATASET: elpv
###########################################################################################################
class elpv(data.Dataset):
    """Dataset class for elpv"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/elpv-dataset-master'):
        self.root_dir = root_dir
        
        self.label_path = root_dir+'/labels.csv'
        self.informations = pd.read_csv(self.label_path)
        self.samples = self.load_samples()
        self.label_list = self.samples['probs'].unique().tolist()
        
    def len(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        prob = self.samples.loc[idx, 'probs']
        type = self.samples.loc[idx, 'types']
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label
        
    def load_samples(self):
        data = np.genfromtxt(self.label_path, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type'])
        image_fnames = np.char.decode(data['path'])
        probs = data['probability']
        types = np.char.decode(data['type'])
        img_classes = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
        img_label = [img_classes.index(prob) for prob in probs]
        
        sample_list = zip(img_label, probs, types, image_fnames)
        df = pd.DataFrame(sample_list, columns=['Image_Label', 'probs', 'types', 'Image_Path'])
        df['Image_Path'] = './My_Datasets/Classification/elpv-dataset-master/' + df['Image_Path']
        return df

    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/elpv.csv")


###########################################################################################################
# DATASET:Magnetic_Tile
###########################################################################################################
class Magnetic_Tile(data.Dataset):
    """Dataset class for Magnetic_Tile"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/Magnetic-Tile-Defect'):
        self.root_dir = root_dir
        self.samples = self.load_samples()
        self.label_list = self.samples["Image_Class"].unique().tolist()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label
    
    def load_samples(self):
        sample_list = []
        file_list = os.listdir(self.root_dir)
        file_list = [filename for filename in file_list if filename[:2]=='MT']
        for file_class in file_list:
            for filename in os.listdir(self.root_dir + '/' + file_class + '/Imgs'):
                if filename[-3:] == 'jpg':
                    # image_path
                    img_path = os.path.join(self.root_dir + '/' + file_class + '/Imgs', filename)
                    # image_class
                    img_class = file_class
                    # image_label
                    img_classes = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']
                    img_label = img_classes.index(img_class)
                    
                    image_item = [img_label, img_class, img_path]
                    sample_list.append(image_item)
        
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])                


    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/Magnetic_Tile.csv")


###########################################################################################################
# DATASET:PCB-200
###########################################################################################################
class PCB_200(data.Dataset):
    """Dataset class for PCB-200"""
    
    def __init__(self, root_dir="./My_Datasets/Classification/PCB-200"):
        self.root_dir = root_dir
        self.samples = self.load_samples()
        self.label_list = self.samples["Image_Class"].unique().tolist()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label
        
    def load_samples(self):
        sample_list = []
        file_list = os.listdir(self.root_dir)
        for file_class in file_list:
            for filename in os.listdir(os.path.join(self.root_dir, file_class)):
                if filename[-3:] == 'bmp' or filename[-3:] == 'png':
                    # image_path
                    img_path = os.path.join(self.root_dir, *[file_class, filename])
                    # image_class
                    img_class = file_class
                    # image_label
                    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
                    img_label = img_classes.index(img_class)
                    
                    # image_item
                    image_item = [img_label, img_class, img_path]
                    sample_list.append(image_item)
                    
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])
    
    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/PCB_200.csv")
                    
            
###########################################################################################################
# DATASET:PCB-Crop
###########################################################################################################
class PCB_Crop(data.Dataset):
    """Dataset class for PCB-Corp"""
    
    def __init__(self, root_dir="./My_Datasets/Classification/PCB-Crop"):
        self.root_dir = root_dir
        self.samples = self.load_samples()
        self.label_list = self.samples["Image_Class"].unique().tolist()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label
        
    def load_samples(self):
        sample_list = []
        file_list = os.listdir(self.root_dir)
        file_list.remove("Missing_hole")
        for file_class in file_list:
            for filename in os.listdir(os.path.join(self.root_dir, file_class)):
                if filename[-3:] == 'bmp' or filename[-3:] == 'png' or filename[-3:] == 'jpg':
                    # image_path
                    img_path = os.path.join(self.root_dir, *[file_class, filename])
                    # image_class
                    img_class = file_class
                    # image_label
                    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
                    img_label = img_classes.index(img_class)
                    
                    # image_item
                    image_item = [img_label, img_class, img_path]
                    sample_list.append(image_item)
                    
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])
    
    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/PCB_Crop.csv")
        

###########################################################################################################
# DATASET:DeepPCB-Crop
###########################################################################################################
class DeepPCB_Crop(data.Dataset):
    """Dataset class for DeepPCB-Corp"""
    
    def __init__(self, root_dir="./My_Datasets/Classification/DeepPCB-Crop"):
        self.root_dir = root_dir
        self.samples = self.load_samples()
        self.label_list = self.samples["Image_Class"].unique().tolist()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        img_label = self.samples.loc[idx, 'Image_Label']
        return img_content, img_label
        
    def load_samples(self):
        sample_list = []
        file_list = os.listdir(self.root_dir)
        file_list.remove("Pin_hole")
        for file_class in file_list:
            for filename in os.listdir(os.path.join(self.root_dir, file_class)):
                if filename[-3:] == 'bmp' or filename[-3:] == 'png' or filename[-3:] == 'jpg':
                    # image_path
                    img_path = os.path.join(self.root_dir, *[file_class, filename])
                    # image_class
                    img_class = file_class
                    # image_label
                    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Pin_hole']
                    img_label = img_classes.index(img_class)
                    
                    # image_item
                    image_item = [img_label, img_class, img_path]
                    sample_list.append(image_item)
                    
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])
    
    def save_csv(self):
        self.samples.to_csv("./My_TAOD/dataset/DeepPCB_Crop.csv")
        
        
###########################################################################################################
# DATASET: Add other dataset
###########################################################################################################

###########################################################################################################
# FUNCTION: dataset_by_label
###########################################################################################################
def dataset_by_label(df, filepath, mode):
    label_list = df['Image_Label'].unique().tolist() 
    os.makedirs(filepath + '/' + mode, exist_ok=True)
    for label in label_list:
        label_dir = filepath + '/' + mode + '/' + str(label) + '.csv'
        df_class = df.loc[df["Image_Label"] == label]
        df_class.to_csv(label_dir)

###########################################################################################################
# FUNCTION: bulid_dataset
# 将原始数据集每类抽样划分为适用于fewshot的训练集和测试集，并存为对应的csv文件
# 之后的Dataloader将直接从对应的csv文件中读取数据的地址
###########################################################################################################
target_dir = '/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset'

def build_dataset(dataset):
    # 重要说明：在dataset_train_size_list中
    # 传入整数：代表训练集尺寸
    # 传入小数：代表训练集所占比例
    if dataset == 'NEU_CLS':
        dataset_origin = NEU_CLS()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 0.7]
        
    elif dataset == 'elpv':
        dataset_origin = elpv()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 0.7]
    
    elif dataset == 'Magnetic_Tile':
        dataset_origin = Magnetic_Tile()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 0.7]
        
    elif dataset == 'PCB_200':
        dataset_origin = PCB_200()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 160, 0.7, [0.6, 0.2, 0.2]]
        
    elif dataset == 'PCB_Crop':
        dataset_origin = PCB_Crop()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 160, 0.7, [0.6, 0.2, 0.2]]
    
    elif dataset == 'DeepPCB_Crop':
        dataset_origin = DeepPCB_Crop()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [5, 10, 30, 50, 160, 0.7, [0.6, 0.2, 0.2]]
        
    else:
        sys.exit(f"ERROR:\t({__name__}):The dataset '{dataset}' doesn't exist")
    
    dataset_origin.save_csv()
    
    dir_dataset = target_dir+'/'+dataset
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset, exist_ok=True)
    
    for dataset_train_size in dataset_train_size_list:
        if isinstance(dataset_train_size, list):
            assert len(dataset_train_size) == 3
            dir_class = dir_dataset + f'/{dataset_train_size[0]}-{dataset_train_size[1]}-{dataset_train_size[2]}-shot'
            os.makedirs(dir_class, exist_ok=True)
            df_train = pd.DataFrame()
            df_validation = pd.DataFrame()
            df_test = pd.DataFrame()
            for label in label_list:
                # 取出标签为label的所有样本
                df_class = df.loc[df["Image_Label"] == label]
                
                if isinstance(dataset_train_size[0], int):
                    samples_size = dataset_train_size[0]
                elif isinstance(dataset_train_size[0], float):
                    samples_size = int(dataset_train_size[0] * len(df_class))
                sample_choose = df_class.sample(samples_size)
                # 将抽出的样本插入训练集df_train
                df_train = pd.concat([df_train, sample_choose])
                # 在标签为label的样本中取出训练集，剩下的作为保留样本
                sample_rest = df_class.drop(sample_choose.index)
                
                if isinstance(dataset_train_size[1], int):
                    samples_size = dataset_train_size[1]
                elif isinstance(dataset_train_size[1], float):
                    samples_size = int(dataset_train_size[1] * len(df_class))
                sample_choose = sample_rest.sample(samples_size)
                # 将抽出的样本插入训练集df_validation
                df_validation = pd.concat([df_validation, sample_choose])
                sample_rest = sample_rest.drop(sample_choose.index)
                
                # 将保留的样本插入测试集df_test
                df_test = pd.concat([df_test, sample_rest])
                
                df_train.to_csv(dir_class + f'/train.csv')
                dataset_by_label(df_train, dir_class, 'train')
                df_validation.to_csv(dir_class + f'/validation.csv')
                dataset_by_label(df_validation, dir_class, 'validation')
                df_test.to_csv(dir_class + f'/test.csv')
                dataset_by_label(df_test, dir_class, 'test')
                
        else:
            dir_class = dir_dataset + f'/{dataset_train_size}-shot'
            os.makedirs(dir_class, exist_ok=True)
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            for label in label_list:
                # 取出标签为label的所有样本
                df_class = df.loc[df["Image_Label"] == label]
                
                # 在标签为label的样本中抽样
                if isinstance(dataset_train_size, int):
                    samples_size = dataset_train_size
                elif isinstance(dataset_train_size, float):
                    samples_size = int(dataset_train_size * len(df_class))
                sample_choose = df_class.sample(samples_size)
                # 将抽出的样本插入训练集df_train
                df_train = pd.concat([df_train, sample_choose])
                # 在标签为label的样本中取出训练集，作为保留样本
                sample_rest = df_class.drop(sample_choose.index)
                
                # 将保留的样本插入测试集df_test
                df_test = pd.concat([df_test, sample_rest])
                           
                df_train.to_csv(dir_class + f'/train.csv')
                dataset_by_label(df_train, dir_class, 'train')
                df_test.to_csv(dir_class + f'/test.csv')
                dataset_by_label(df_test, dir_class, 'test')
        
    print(f'({__name__}):Dataset:{dataset}\tBuild Successfully!!!')


###########################################################################################################
# 运行函数：创建Dataset
###########################################################################################################
if __name__ == "__main__":
    # build_dataset('NEU_CLS')
    # build_dataset('elpv')
    # build_dataset('Magnetic_Tile')
    # build_dataset('PCB_200')
    # build_dataset('PCB_Crop')
    # build_dataset('DeepPCB_Crop')
    pass
