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
def load_img_for_mmd(csv_path, trans=None, batch_size=50):
    """_summary_

    Args:
        csv_path (str): a csv file with image path
        trans (_type_, optional): Defaults to None.

    Returns:
        [images, category labels]
    """
    df = pd.read_csv(csv_path)
    nums = len(df)
    category_lables = df["Image_Label"]
    image_path = df["Image_Path"]
    
    images = []
    image_list = random.sample(list(image_path), k=batch_size)
    for i, img in enumerate(image_list):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        if trans:
            img = trans(img)
        else:
            img = img.transpose(2, 0, 1)
        images.append(img.cuda())
    
    if trans:
        images = torch.cat(images, dim=0)
    else:
        images = torch.FloatTensor(np.array(images))

    return images, category_lables


#######################################################################################################
#### FUNCTION: load_img_for_fid()
#######################################################################################################
def load_img_for_fid(csv_path, trans=None, batch_size=50):
    df = pd.read_csv(csv_path)
    
    nums = len(df)
    category_lables = df["Image_Label"]
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
#### FUNCTION: get_subfolders()
#######################################################################################################
def get_subfolders(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

#######################################################################################################
#### FUNCTION: PFS_baseline_from_scratch_record_data()
#######################################################################################################
def PFS_baseline_from_scratch_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
        
        G_loss = results["G_loss"]
        D_loss = results["D_loss"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_loss.jpg')
        plt.close()


#######################################################################################################
#### FUNCTION: baseline_from_scratch_record_data()
#######################################################################################################
def baseline_from_scratch_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
        
        G_adv_loss = results["G_adv_loss"]
        G_FM_loss = results["G_FM_loss"]
        G_loss = results["G_loss"]
        D_real_adv_loss = results["D_real_adv_loss"]
        D_fake_adv_loss = results["D_fake_adv_loss"]
        D_loss = results["D_loss"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss], label="G_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_FM_loss], label="G_FM_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_FM_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_loss.jpg')
        plt.close()
        
        
#######################################################################################################
#### FUNCTION: PFS_baseline_finetuning_record_data()
#######################################################################################################
def PFS_baseline_finetuning_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + "_from_" + config.G_init_class + ' ' + config.category + ' ' + config.time
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
        
        G_loss = results["G_loss"]
        D_loss = results["D_loss"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_loss.jpg')
        plt.close()
#######################################################################################################
#### FUNCTION: stage1_record_data()
#######################################################################################################
def stage1_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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

        G_loss = results["G_loss"]
        D_loss = results["D_loss"]
        KLD_c = results["KLD_c"]
        KLD_s = results["KLD_s"]
        imgrecon = results["imgrecon"]
        s_recon = results["s_recon"]
        Perceptual_loss = results["Perceptual_loss"]

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in KLD_c], label="KLD_c")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/KLD_c.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in KLD_s], label="KLD_s")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/KLD_s.jpg')
        plt.close()

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in imgrecon], label="imgrecon")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/imgrecon.jpg')
        plt.close()

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in s_recon], label="s_recon")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/s_recon.jpg')
        plt.close()
    
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in Perceptual_loss], label="Perceptual_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/Perceptual_loss.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: stage2_record_data()
#######################################################################################################
def stage2_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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

        G_loss = results["G_loss"]
        D_loss = results["D_loss"]
        KLD_c = results["KLD_c"]
        KLD_s = results["KLD_s"]
        imgrecon = results["imgrecon"]
        s_recon = results["s_recon"]
        Perceptual_loss = results["Perceptual_loss"]

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in KLD_c], label="KLD_c")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/KLD_c.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in KLD_s], label="KLD_s")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/KLD_s.jpg')
        plt.close()

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in imgrecon], label="imgrecon")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/imgrecon.jpg')
        plt.close()

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in s_recon], label="s_recon")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/s_recon.jpg')
        plt.close()
    
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in Perceptual_loss], label="Perceptual_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/Perceptual_loss.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: cogan_record_data()
#######################################################################################################
def cogan_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_S_class + '_2_' + config.dataset_T_class + ' ' + config.category + ' ' + config.time
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
        S_D_loss = results["S_D_loss"]
        T_D_loss = results["T_D_loss"]
        S_G_loss = results["S_G_loss"]
        T_G_loss = results["T_G_loss"]


        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in S_D_loss], label="S_D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/S_D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in T_D_loss], label="T_D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/T_D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in S_G_loss], label="S_G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/S_G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in T_G_loss], label="T_G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/T_G_loss.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: cyclegan_record_data()
#######################################################################################################
def cyclegan_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_S_class + '_2_' + config.dataset_T_class + ' ' + config.category + ' ' + config.time
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
        loss_G = results["loss_G"]
        loss_G_identity = results["loss_G_identity"]
        loss_G_GAN = results["loss_G_GAN"]
        loss_G_cycle = results["loss_G_cycle"]
        loss_D = results["loss_D"]


        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in loss_G], label="loss_G")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/loss_G.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in loss_G_identity], label="loss_G_identity")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/loss_G_identity.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in loss_G_GAN], label="loss_G_GAN")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/loss_G_GAN.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in loss_G_cycle], label="loss_G_cycle")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/loss_G_cycle.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in loss_D], label="loss_D")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/loss_D.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: S_trainer_record_data()
#######################################################################################################
def S_trainer_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.category + ' ' + config.time
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
    filename = config.dataset_target + ' ' + config.category + ' ' + config.time
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
#### FUNCTION: DCGAN_record_data()
#######################################################################################################
def DCGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
            
        D_real_adv_loss = results["D_real_adv_loss"]
        D_fake_adv_loss = results["D_fake_adv_loss"]
        D_loss = results["D_loss"]    
        G_loss = results["G_loss"]
             
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_real_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()


#######################################################################################################
#### FUNCTION: WGAN_GP_record_data()
#######################################################################################################
def WGAN_GP_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
            
        D_real_loss = results["D_real_loss"]
        D_fake_loss = results["D_fake_loss"]
        D_gp = results["D_gp"]
        D_loss = results["D_loss"]    
        G_loss = results["G_loss"]
             
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_loss], label="D_real_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_real_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_fake_loss], label="D_fake_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_fake_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_gp], label="D_gp")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_gp.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_loss], label="D_real_loss", color="tab:red")
        ax1.plot([y for y in D_fake_loss], label="D_fake_loss", color="tab:blue")
        ax1.plot([y for y in D_gp], label="D_gp", color="tab:green")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()


#######################################################################################################
#### FUNCTION: SAGAN_record_data()
#######################################################################################################
def SAGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
            
        D_real_adv_loss = results["D_real_adv_loss"]
        D_fake_adv_loss = results["D_fake_adv_loss"]
        D_loss = results["D_loss"]    
        G_loss = results["G_loss"]
             
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_real_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()
        
        
#######################################################################################################
#### FUNCTION: ConGAN_record_data()
#######################################################################################################
def ConGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        G_fake_adv_loss = results["G_fake_adv_loss"]
        G_FM_loss = results["G_FM_loss"]
        G_Triplet_loss = results["G_Triplet_loss"]
        G_loss = results["G_loss"]
        
        D_real_adv_loss = results["D_real_adv_loss"]
        D_fake_adv_loss = results["D_fake_adv_loss"]
        D_loss = results["D_loss"]    
        
             
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_fake_adv_loss], label="G_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_FM_loss], label="G_FM_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_FM_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_Triplet_loss], label="G_Triplet_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_Triplet_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_fake_adv_loss], label="G_fake_adv_loss", color="tab:red")
        ax1.plot([y for y in G_FM_loss], label="G_FM_loss", color="tab:blue")
        ax1.plot([y for y in G_Triplet_loss], label="G_Triplet_loss", color="tab:green")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_real_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss.jpg')
        plt.close()


#######################################################################################################
#### FUNCTION: CoGAN_record_data()
#######################################################################################################
def CoGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        D_real_adv_loss = results["D_real_adv_loss"]
        D_fake_adv_loss = results["D_fake_adv_loss"]
        D_mse_loss = results["D_mse_loss"]
        D_cls_loss = results["D_cls_loss"]
        D_loss = results["D_loss"]    
        
        G_fake_adv_loss = results["G_fake_adv_loss"]
        G_loss = results["G_loss"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_real_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_mse_loss], label="D_mse_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_mse_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_cls_loss], label="D_cls_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_cls_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_real_adv_loss], label="D_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_fake_adv_loss], label="D_fake_adv_loss", color="tab:blue")
        ax1.plot([y for y in D_mse_loss], label="D_mse_loss", color="tab:green")
        ax1.plot([y for y in D_cls_loss], label="D_cls_loss", color="tab:pink")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_loss], label="D_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_fake_adv_loss], label="G_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_fake_adv_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()


#######################################################################################################
#### FUNCTION: CycleGAN_record_data()
#######################################################################################################
def CycleGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        G_adv_loss_Src2Tar = results["G_adv_loss_Src2Tar"]
        G_adv_loss_Tar2Src = results["G_adv_loss_Tar2Src"]
        G_cycle_loss_Src2Tar2Src = results["G_cycle_loss_Src2Tar2Src"]
        G_cycle_loss_Tar2Src2Tar = results["G_cycle_loss_Tar2Src2Tar"]
        G_identity_loss_Src = results["G_identity_loss_Src"]
        G_identity_loss_Tar = results["G_identity_loss_Tar"]
        G_loss = results["G_loss"]
        
        D_Src_real_adv_loss = results["D_Src_real_adv_loss"]
        D_Src_fake_adv_loss = results["D_Src_fake_adv_loss"]
        D_Src_loss = results["D_Src_loss"]
        
        D_Tar_real_adv_loss = results["D_Tar_real_adv_loss"]
        D_Tar_fake_adv_loss = results["D_Tar_fake_adv_loss"]
        D_Tar_loss = results["D_Tar_loss"]
        
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss_Src2Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss_Tar2Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar", color="tab:red")
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Src2Tar2Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Tar2Src2Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src", color="tab:red")
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss_Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss_Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src", color="tab:red")
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar", color="tab:red")
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src", color="tab:blue")
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src", color="tab:green")
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar", color="tab:purple")
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src", color="tab:cyan")
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar", color="tab:brown")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_loss], label="D_Src_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_loss], label="D_Tar_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: MyGAN_record_data()
#######################################################################################################
def MyGAN_record_data(save_dir, content, flag_plot=True):
    """Save Data"""
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        G_adv_loss = results["G_adv_loss"]
        G_FM_loss = results["G_FM_loss"]
        G_cycle_loss_Src = results["G_cycle_loss_Src"]
        G_cycle_loss_Tar = results["G_cycle_loss_Tar"]
        G_loss = results["G_loss"]
        
        D_Src_real_adv_loss = results["D_Src_real_adv_loss"]
        D_Src_fake_adv_loss = results["D_Src_fake_adv_loss"]
        D_Src_recovery_adv_loss = results["D_Src_recovery_adv_loss"]
        D_Src_loss = results["D_Src_loss"]
        
        D_Tar_real_adv_loss = results["D_Tar_real_adv_loss"]
        D_Tar_fake_adv_loss = results["D_Tar_fake_adv_loss"]
        D_Tar_recovery_adv_loss = results["D_Tar_recovery_adv_loss"]
        D_Tar_loss = results["D_Tar_loss"]
        
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss], label="G_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_FM_loss], label="G_FM_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_FM_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src], label="G_cycle_loss_Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Tar], label="G_cycle_loss_Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src], label="G_cycle_loss_Src", color="tab:red")
        ax1.plot([y for y in G_cycle_loss_Tar], label="G_cycle_loss_Tar", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss], label="G_adv_loss", color="tab:red")
        ax1.plot([y for y in G_FM_loss], label="G_FM_loss", color="tab:green")
        ax1.plot([y for y in G_cycle_loss_Src], label="G_cycle_loss_Src", color="tab:purple")
        ax1.plot([y for y in G_cycle_loss_Src], label="G_cycle_loss_Src", color="tab:cyan")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_recovery_adv_loss], label="D_Src_recovery_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_recovery_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss", color="tab:blue")
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_recovery_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_loss], label="D_Src_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_recovery_adv_loss], label="D_Tar_recovery_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_recovery_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss", color="tab:blue")
        ax1.plot([y for y in D_Tar_recovery_adv_loss], label="D_Tar_recovery_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_loss], label="D_Tar_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss.jpg')
        plt.close()

#######################################################################################################
#### FUNCTION: UNIT_record_data()
#######################################################################################################
def UNIT_record_data(save_dir, content, flag_plot=True):
    """Save Data"""   
    filepath = f"{save_dir}/results/data.csv"
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        G_adv_loss_Src2Tar = results["G_adv_loss_Src2Tar"]
        G_adv_loss_Tar2Src = results["G_adv_loss_Tar2Src"]
        G_cycle_loss_Src2Tar2Src = results["G_cycle_loss_Src2Tar2Src"]
        G_cycle_loss_Tar2Src2Tar = results["G_cycle_loss_Tar2Src2Tar"]
        G_identity_loss_Src = results["G_identity_loss_Src"]
        G_identity_loss_Tar = results["G_identity_loss_Tar"]
        G_loss = results["G_loss"]
        
        D_Src_real_adv_loss = results["D_Src_real_adv_loss"]
        D_Src_fake_adv_loss = results["D_Src_fake_adv_loss"]
        D_Src_loss = results["D_Src_loss"]
        
        D_Tar_real_adv_loss = results["D_Tar_real_adv_loss"]
        D_Tar_fake_adv_loss = results["D_Tar_fake_adv_loss"]
        D_Tar_loss = results["D_Tar_loss"]
        
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss_Src2Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss_Tar2Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar", color="tab:red")
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_adv_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Src2Tar2Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss_Tar2Src2Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src", color="tab:red")
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_cycle_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss_Src.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss_Tar.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src", color="tab:red")
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_identity_loss.jpg')
        plt.close()
        # 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_loss], label="G_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in G_adv_loss_Src2Tar], label="G_adv_loss_Src2Tar", color="tab:red")
        ax1.plot([y for y in G_adv_loss_Tar2Src], label="G_adv_loss_Tar2Src", color="tab:blue")
        ax1.plot([y for y in G_cycle_loss_Src2Tar2Src], label="G_cycle_loss_Src2Tar2Src", color="tab:green")
        ax1.plot([y for y in G_cycle_loss_Tar2Src2Tar], label="G_cycle_loss_Tar2Src2Tar", color="tab:purple")
        ax1.plot([y for y in G_identity_loss_Src], label="G_identity_loss_Src", color="tab:cyan")
        ax1.plot([y for y in G_identity_loss_Tar], label="G_identity_loss_Tar", color="tab:brown")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/G_loss_all.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_real_adv_loss], label="D_Src_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Src_fake_adv_loss], label="D_Src_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Src_loss], label="D_Src_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Src_loss.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_real_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_fake_adv_loss.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_real_adv_loss], label="D_Tar_real_adv_loss", color="tab:red")
        ax1.plot([y for y in D_Tar_fake_adv_loss], label="D_Tar_fake_adv_loss", color="tab:blue")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss_all.jpg')
        plt.close()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_Tar_loss], label="D_Tar_loss")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/results/D_Tar_loss.jpg')
        plt.close()

#######################################################################################################
# Function Test
#######################################################################################################
if __name__ == "__main__":
    pass