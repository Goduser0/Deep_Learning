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
from TA_VAE import VAE
from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchDiscriminator
from TA_layers import KLDLoss

#######################################################################################################
## FUNCTIONS: train_VAE()
#######################################################################################################
def train_VAE():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    trans = T.Compose([T.ToTensor(), T.Resize((128, 128))])
    dataloader = get_loader("PCB_200", 
                            "./My_TAOD/dataset/PCB_200/1.0-shot/train/0.csv", 
                            32, 
                            4, 
                            shuffle=True, 
                            trans=trans,
                            img_type='ndarray',
                            drop_last=False
                            ) # 像素值范围：（-1, 1）[B, C, H, W]
    
    vae_test = VAE(3, 128).cuda()
    vae_optim = torch.optim.Adam(vae_test.parameters(), lr=1e-5, betas=[0.0, 0.9])
    
    G = FeatureMatchGenerator(
        3, 128, 128, 64, 1e-5
    ).cuda()
    g_reg_ratio = 4 / (4 + 1)
    g_optim = torch.optim.Adam(
        G.parameters(),
        lr = 1e-5 * g_reg_ratio,
        betas=(0.0 ** g_reg_ratio, 0.9 ** g_reg_ratio),
)   
    
    num_epochs = 2000
    for epoch in tqdm.trange(num_epochs):
        vae_test.train()
        G.train()
        
        epoch_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        for i, data in enumerate(dataloader):
            real_imgs = data[0].expand(-1, 3, -1, -1)
            real_imgs = real_imgs.cuda()
            
            [_, _, mu, log_var, z] = vae_test(real_imgs)
            fake_imgs = G(z)
            
            loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0) #batchmean
            loss_recon = F.mse_loss(real_imgs, fake_imgs) #batchmean
            loss_total = 0.5 * loss_kld + 10.0 * loss_recon
            
            epoch_loss += loss_total.item()
            kl_loss += loss_kld.item()
            recon_loss += loss_recon.item() 
            
            vae_optim.zero_grad()
            g_optim.zero_grad()
            loss_total.backward()
            vae_optim.step()
            g_optim.step()
            
        print(f"[{epoch+1}/{num_epochs}/{len(dataloader)}] [loss:{epoch_loss / len(dataloader):.4f}] [recon_loss:{recon_loss / len(dataloader):.4f}] [kld_loss:{kl_loss / len(dataloader):.4f}]")
        
        with torch.no_grad():
            z = torch.randn(1, 128).cuda()
            images = G(z)
            images = images.detach().to('cpu').numpy()[0]
            images = np.transpose(images, (1, 2, 0))
            plt.imshow(images)
            plt.savefig("Train_VAE_gen.png")
            plt.close()
        
    torch.save(vae_test.state_dict(), 'vae_pcb_with_G.pth')
    
if __name__ == '__main__':
    train_VAE()