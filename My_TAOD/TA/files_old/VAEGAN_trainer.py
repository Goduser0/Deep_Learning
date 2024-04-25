import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torch.nn.functional as F

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255, img_255to1
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_VAE import VariationalAutoEncoder
from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchDiscriminator
from TA_layers import KLDLoss, PerceptualLoss
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_utils import generator_adv_loss, discriminator_adv_loss

#######################################################################################################
## FUNCTIONS: train_VAEGAN()
#######################################################################################################
def train_VAEGAN():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    trans = T.Compose([T.ToTensor(), T.Resize((128, 128))])
    dataloader = get_loader(
        "PCB_200",
        "./My_TAOD/dataset/PCB_200/1.0-shot/train/0.csv",
        32,
        4,
        shuffle=True,
        trans=trans,
        img_type="ndarray",
        drop_last=False,
    )
    
    VAE = VariationalAutoEncoder(
        in_channels=3,
        latent_dim=128,
    ).cuda()
    VAE_optim = torch.optim.Adam(VAE.parameters(), lr=1e-4, betas=[0.0, 0.9])
    
    G = FeatureMatchGenerator(
        n_mlp=3,
        img_size=128,
        z_dim=128,
        conv_dim=64,
        lr_mlp=1e-4
    ).cuda()
    G_reg_ratio = 4.0 / (4.0 + 1.0)
    G_optim = torch.optim.Adam(G.parameters(), lr=1e-3 * G_reg_ratio, betas=[0.0 ** G_reg_ratio, 0.9 ** G_reg_ratio])
    
    D = FeatureMatchDiscriminator(
        img_size=128,
        conv_dim=64,
    ).cuda()
    D_reg_ratio = 4.0 / (4.0 + 1.0)
    D_optim = torch.optim.Adam(D.parameters(), lr=1e-4 * D_reg_ratio, betas=[0.0 ** D_reg_ratio, 0.9 ** D_reg_ratio])
    
    Cal_PLloss = PerceptualLoss(torch.device("cuda:0"))
    
    num_epochs = 2000
    for epoch in tqdm.tqdm(range(1, 2000+1), desc=f"On training"):
        VAE.train()
        G.train()
        D.train()
        
        epochloss_VAE = 0.0
        epochloss_G = 0.0
        epochloss_D = 0.0
        for i, data in enumerate(dataloader):
            real_imgs = data[0].cuda()
            catagory_labels = data[1].cuda()
    
            # Train VAE&G
            VAE_optim.zero_grad()
            G_optim.zero_grad()
            
            [vae_recon_imgs, _, mu, log_var, vae_z] = VAE(real_imgs)
            fake_imgs = G(vae_z)
            
            real_imgs_out = D(real_imgs)
            fake_imgs_out = D(fake_imgs)
            
            # VAE_loss
            VAE_kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
            VAE_recon_loss = F.mse_loss(real_imgs, fake_imgs)
            
            VAE_total_loss = VAE_kld_loss * 0.1 + VAE_recon_loss
            
            # G_loss
            G_adv_loss = generator_adv_loss("FeatureMatchGenerator", fake_imgs_out[-2])
            
            criterionFeat = torch.nn.L1Loss()
            G_fm_loss = torch.cuda.FloatTensor(1).fill_(0)
            num_feature_maps = 4
            lambda_fm_weight = [1.0, 1.0, 1.0, 1.0]
            for i_map in range(num_feature_maps):
                fm_loss = criterionFeat(fake_imgs_out[i_map], real_imgs_out[i_map])
                G_fm_loss += (fm_loss) * lambda_fm_weight[i_map]
            
            G_total_loss = G_adv_loss * 0.1 + G_fm_loss
            
            (VAE_total_loss + G_total_loss).backward()

            VAE_optim.step()
            G_optim.step()
            
            # Train D
            D_optim.zero_grad()
            real_imgs_out = D(real_imgs)
            fake_imgs_out = D(fake_imgs.detach())
            
            # D_loss
            D_total_loss = discriminator_adv_loss("FeatureMatchDiscriminator", fake_imgs_out[-2], real_imgs_out[-2])
            D_total_loss.backward()
            D_optim.step()
            
            epochloss_VAE += VAE_total_loss.item()
            epochloss_G += G_total_loss.item()
            epochloss_D += D_total_loss.item()
            
        print(f"[{epoch}/{num_epochs}/{len(dataloader)}] [VAE:{epochloss_VAE / len(dataloader):.4f}] [G:{epochloss_G / len(dataloader):.4f}] [D:{epochloss_D / len(dataloader):.4f}]")    
        
        with torch.no_grad():
            z = torch.randn(1, 128).cuda()
            images = G(z)
            images = images.detach().to('cpu').numpy()[0]
            images = np.transpose(images, (1, 2, 0))
            plt.imshow(images)
            plt.savefig("Train_VAEGAN.png")
            plt.close()
            
    torch.save(VAE.state_dict(), 'VAEGAN_VAE.pth')
    torch.save(G.state_dict(), 'VAEGAN_G.pth')
    torch.save(D.state_dict(), 'VAEGAN_D.pth')    
            
if __name__ == "__main__":
    train_VAEGAN()
    
    