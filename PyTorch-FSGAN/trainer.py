import torch
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import os
import argparse

# from model import Generator, Discriminator
from model_sphere import Generator, Discriminator
from dataloader import GANDataset, transform, dataLoader
from losses import generator_loss, discriminator_loss, gradient_penalty

from utils.checkpoint import save_loss, save_samples, plot_loss


# get config
parse = argparse.ArgumentParser()
parse.add_argument("--gan_type", type=str, default="wgan-gp", choices=["gan", "wgan-gp", "wgan-lp", "lsgan", "hinge", "dragan", "rgan", "ragan", "sphere"])
parse.add_argument("--data_path", type=str, default="dataset/NEU/NEU-50/train/Cr")
parse.add_argument("--n_epochs", type=int, default=500)
parse.add_argument("--batch_size", type=int, default=32)
parse.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parse.add_argument("--gpu_id", type=int, default=0)
config = parse.parse_args()
print(config)


# model G and D, choose GPU
gpu_id = str(config.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gan_type = config.gan_type
G = Generator()
D = Discriminator()
G.cuda()
D.cuda()


# data loader
trans = transform(resize=True, totensor=True, normalize=True)  # (B, C, H, W)
data_path = config.data_path
dataloader = dataLoader(data_path, transform=trans, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=2)


# optimizers, SAGAN TTUR
g_lr = 0.0001
d_lr = 0.0004
beta1 = 0.0
beta2 = 0.9
opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=[beta1, beta2])
opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=[beta1, beta2])


# checkpoint and sample path config
if config.batch_size == 30:
    checkpoint_path = "../work_dir/checkpoints_v2_30/" + gan_type + "/" + data_path.split('/')[-1] + "/" + datetime.now().strftime("%Y%m%d%H%M")
if config.batch_size == 10:
    checkpoint_path = "../work_dir/checkpoints_v2_10/" + gan_type + "/" + data_path.split('/')[-1] + "/" + datetime.now().strftime("%Y%m%d%H%M")
sample_path = checkpoint_path + "/samples"
sample_interval = config.sample_interval


# =============
# train
# =============
Tensor = torch.cuda.FloatTensor
n_epochs = config.n_epochs  # depends on batch_size and samples num in dataset
z_dim = 128

for epoch in range(1, n_epochs + 1):
    for i, imgs in enumerate(dataloader):
        # configs input
        real_imgs = Variable(imgs, requires_grad=False).cuda()  # ->torch.float32
        
        # -----------------
        # train generator
        # -----------------
        opt_G.zero_grad()
        
        # generate fake images, z_dim = 128
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 128))), requires_grad=False).cuda()
        fake_imgs = G(z)
        
        # Generator loss
        fake_images_out = D(fake_imgs)
        if gan_type == "rgan" or gan_type == "ragan":
            real_images_out = D(real_imgs.detach())
            g_loss = generator_loss(gan_type, fake_images_out, real_images_out)
        else:
            g_loss = generator_loss(gan_type, fake_images_out)

        g_loss.backward()
        opt_G.step()

        # -----------------
        # train discriminator
        # -----------------
        opt_D.zero_grad()
        
        # Discriminator loss
        real_imgs_out = D(real_imgs)
        fake_imgs_out = D(fake_imgs.detach())

        # Discriminator loss
        lambda_gp = 1.0
        if gan_type == "wgan-gp" or gan_type == "wgan-lp":
            # calculate gp need to input `data`, or the imgs's grad will be calculated twice
            GP = lambda_gp * gradient_penalty(gan_type, D, real_imgs.data, fake_imgs.data)
        else:
            GP = 0
        d_loss = discriminator_loss(gan_type, real_imgs_out, fake_imgs_out) + GP

        d_loss.backward()
        opt_D.step()


        # save loss_log.txt to /work_dir/checkpoints
        save_loss(checkpoint_path, epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())

        # save samples to /work_dir/checkpoints
        save_samples(sample_path, sample_interval, epoch, i, len(dataloader), fake_imgs.data)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()), end='\n'
        )

    # save model Generator
    if epoch % (n_epochs // 25) == 0:
        net_G_path = checkpoint_path + '/%d_net_G.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": G.state_dict(),
            "gan_type": gan_type,
            "dataset": data_path,
            "z_dim": z_dim
        }, net_G_path)
    # save model Discriminator final for Break-Point Training
    if epoch == n_epochs:
        net_D_path = checkpoint_path + '/%d_net_D.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": D.state_dict(),
            "gan_type": gan_type,
            "dataset": data_path,
        }, net_D_path)

# plot loss
plot_loss(checkpoint_path + '/loss_log.txt', mode="epoch")
plot_loss(checkpoint_path + '/loss_log.txt', mode="batch")
