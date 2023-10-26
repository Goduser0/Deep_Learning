import os
import random
import argparse
import time

import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T

#######################################################################################################
#### FUNCTION: requires_grad()
#######################################################################################################
def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


#######################################################################################################
#### FUNCTION: plt_tsne()
#######################################################################################################
def plt_tsne2d(X, Y, labels=None):
    # tsne
    tsne = TSNE(n_components=2, random_state=501)
    X_tsne = tsne.fit_transform(X)
    # 归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_norm = scaler.fit_transform(X_tsne)
    
    # plot
    plt.figure(figsize=(15, 15))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=Y, s=50)
    if labels:
        pass
    else:
        labels = range(len(scatter.legend_elements()[0]))
    plt.legend(handles=scatter.legend_elements()[0],labels=labels,title="classes")
                
    t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_path = "./My_TAOD/TA/results/" + str(t) + ".png"
    plt.savefig(save_path)
    plt.close()


#######################################################################################################
#### FUNCTION: load_img_for_mmd()
#######################################################################################################
def load_img_for_mmd(csv_path, trans=None):
    """_summary_

    Args:
        csv_path (str): a csv file with image path
        trans (_type_, optional): Defaults to None.

    Returns:
        [images, catagory labels]
    """
    df = pd.read_csv(csv_path)
    catagory_lables = df["Image_Label"]
    image_path = df["Image_Path"]
    
    images = []
    for i in image_path:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        if trans:
            img = trans(img)
        else:
            img = img.transpose(2, 0, 1)
        images.append(img)
    
    if trans:
        images = torch.cat(images, dim=0)
    else:
        images = torch.FloatTensor(np.array(images))

    return images, catagory_lables


#######################################################################################################
#### FUNCTION: load_img_for_fid()
#######################################################################################################
def load_img_for_fid(csv_path, trans=None, batch_size=25):
    df = pd.read_csv(csv_path)
    
    nums = len(df)
    catagory_lables = df["Image_Label"]
    image_path = df["Image_Path"]

    images = []
    image_list = random.sample(list(image_path), k=batch_size)
    for i, img in enumerate(image_list):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        if trans:
            img = T.ToPILImage()(img)
            img = trans(img)
    
        images.append(img.numpy())
    images = torch.FloatTensor(np.array(images))
    
    return images


#######################################################################################################
#### FUNCTION: generator_adv_loss()
#######################################################################################################
def generator_adv_loss(generator_type :str, fake_labels, real_labels=None):
    """ 
    Provide generator type to return adversarial losses 
    G_type: ["FeatureMatchGenerator"]
    """
    
    if generator_type.lower() == "featurematchgenerator":
        fake_loss = torch.nn.BCEWithLogitsLoss()(fake_labels, torch.ones_like(fake_labels))
        g_loss = fake_loss
    else:
        raise KeyError(f"{generator_type} is not exist")
    
    return g_loss


#######################################################################################################
#### FUNCTION: discriminator_adv_loss()
#######################################################################################################
def discriminator_adv_loss(discriminator_type :str, fake_labels, real_labels):
    """
    Provide discriminator type to return adversarial losses
    D_type: ["FeatureMatchDiscriminator"]
    """
    if discriminator_type.lower() == "featurematchdiscriminator":
        real_loss = torch.nn.BCEWithLogitsLoss()(real_labels, torch.ones_like(real_labels))
        fake_loss = torch.nn.BCEWithLogitsLoss()(fake_labels, torch.zeros_like(fake_labels))
        d_loss = (real_loss + fake_loss) / 2.0
    else:
        raise KeyError(f"{discriminator_type} is not exist")
    
    return d_loss

#######################################################################################################
#### FUNCTION: S_trainer_record_data()
#######################################################################################################
def S_trainer_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.catagory + ' ' + config.time
    filepath = config.results_dir + '/' + filename + '.csv'
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    if not os.path.exists(f"{config.results_dir}/{filename}"):
        os.makedirs(f"{config.results_dir}/{filename}")
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        epoch = results["epoch"]
        num_epochs = results["num_epochs"]
        batch = results["batch"]
        num_batchs = results["num_batchs"]
        
        TotalLoss_vae = results["TotalLoss_vae"]
        TotalLoss_g = results["TotalLoss_g"]
        D_TotalLoss_g = results["D_TotalLoss_g"]
        D_TotalLoss_d = results["D_TotalLoss_d"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_vae], label="TotalLoss_vae")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_vae.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_g], label="TotalLoss_g")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_g.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_TotalLoss_g], label="D_TotalLoss_g")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_TotalLoss_g.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_TotalLoss_d], label="D_TotalLoss_d")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_TotalLoss_d.jpg')
        plt.close()
        

#######################################################################################################
#### FUNCTION: S2T_trainer_record_data()
#######################################################################################################
def S2T_trainer_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_target + ' ' + config.catagory + ' ' + config.time
    filepath = config.results_dir + '/' + filename + '.csv'
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    if not os.path.exists(f"{config.results_dir}/{filename}"):
        os.makedirs(f"{config.results_dir}/{filename}")
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        epoch = results["epoch"]
        num_epochs = results["num_epochs"]
        
        TotalLoss_vae_unique = results["TotalLoss_vae_unique"]
        TotalLoss_g_target = results["TotalLoss_g_target"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_vae_unique], label="TotalLoss_vae_unique")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_vae_unique.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_g_target], label="TotalLoss_g_target")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_g_target.jpg')
        plt.close()
        
    
#######################################################################################################
# Function Test
#######################################################################################################
def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subspace_std", default=0.1)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--n_sample", default=120)
    parser.add_argument("--latent_dim", default=10)
    config = parser.parse_args()

    df = pd.read_csv("./My_TAOD/dataset/PCB_Crop/0.7-shot/train.csv")
    image_path = df["Image_Path"]
    image_label = df["Image_Label"]
    image_path = list(image_path)

    image = []
    for path in image_path:
        img = Image.open(path).convert('RGB')
        img_array = np.asarray(img)
        trans = T.Compose([T.ToTensor(), T.Resize((224, 224))])
        img_array = trans(img_array)
        image.append(img_array)
    
    image = torch.stack(image)
    image = image.view(len(image), -1)
    image_label = torch.Tensor(image_label)
    plt_tsne2d(image, image_label)
    
if __name__ == "__main__":
    test()