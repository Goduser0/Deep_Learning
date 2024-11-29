import torch
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import os
import argparse
from model_fsgan_v4 import Generator, Discriminator
from dataloader import dataLoader
from losses import generator_loss, discriminator_loss
from utils.checkpoint import save_samples
from utils.util import img_uint2tensor, img_tensor2uint, save_loss, plot_loss, gradient_penalty
from augmentation import ImgAugmentation
from utils.triplet_loss import MMD_TripletLoss, COSSIM_TripletLoss, HYPERSPHERE_TripletLoss


# get config
parse = argparse.ArgumentParser()
parse.add_argument("--gan_type", type=str, default="wgan-gp", choices=["gan", "wgan-gp", "wgan-lp", "lsgan", "hinge", "dragan", "rgan", "ragan", "sphere"])
parse.add_argument("--data_path", type=str, default="dataset/NEU/NEU-50-r64/train/Cr")
parse.add_argument("--n_epochs", type=int, default=500)
parse.add_argument("--batch_size", type=int, default=32)
parse.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parse.add_argument("--gpu_id", type=int, default=0)
parse.add_argument("--margin", type=float, default=0)
parse.add_argument("--lambda_gan_g", type=float, default=0.05)
parse.add_argument("--name", type=str, default="GAN_G5e-2_FMLattn_HYPTripw10M05MLPProv2")
parse.add_argument("--workdir", type=str, default="work_dir_pcb2")

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
data_path = config.data_path
dataloader = dataLoader(data_path, transform=None, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=2)


# optimizers, SAGAN TTUR
g_lr = 0.0001
d_lr = 0.0004 
beta1 = 0.0
beta2 = 0.9
opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=[beta1, beta2])
opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=[beta1, beta2])


# checkpoint and sample path config
# checkpoint_path = "work_dir/checkpoints_v4/" + gan_type + "/" + data_path.split('/')[-1] + "/" + datetime.now().strftime("%Y%m%d%H%M")

checkpoint_path = config.workdir + "/checkpoints/" + gan_type + "/" + data_path.split('/')[-1] + "/" + config.name

sample_path = checkpoint_path + "/samples"
sample_interval = config.sample_interval


# =============
# train
# =============
Tensor = torch.cuda.FloatTensor
n_epochs = config.n_epochs  # depends on batch_size and samples num in dataset
z_dim = 128
aug = ImgAugmentation()


for epoch in range(1, n_epochs + 1):
    for i, imgs in enumerate(dataloader):
        # configs input
        real_imgs_uint = Variable(imgs, requires_grad=False)  # [N, H, W, C], 0-255, cpu
        real_imgs_uint_aug = aug.imgRealAugment(real_imgs_uint)  # [N, H, W, C], 0-255, cpu
        
        real_imgs = img_uint2tensor(real_imgs_uint).cuda()  # [N, C, H, W], -1-1, cuda
        real_imgs_aug = img_uint2tensor(real_imgs_uint_aug).cuda()  # [N, C, H, W], -1-1, cuda

        # -----------------
        # train generator
        # -----------------
        opt_G.zero_grad()

        # generate fake images, z_dim = 128
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 128))), requires_grad=False).cuda()
        fake_imgs = G(z)  # [N, C, H, W], -1-1, gpu
        fake_imgs_uint = img_tensor2uint(fake_imgs)  # [N, H, W, C], 0-255, cuda
        fake_imgs_uint_aug = aug.imgFakeAugment(fake_imgs_uint.cpu())  # [N, H, W, C], 0-255, cpu
        fake_imgs_aug = img_uint2tensor(fake_imgs_uint_aug).cuda()  # [N, C, H, W], -1-1, cuda

        # Calculate Generator loss
        fake_imgs_out = D(fake_imgs)  # [fmap1, fmap2, fmap3, fmap4, final_out, project_out]
        # real_imgs_out = D(real_imgs.detach())
        real_imgs_out = D(real_imgs)

        fake_imgs_aug_out = D(fake_imgs_aug)
        # real_imgs_aug_out = D(real_imgs_aug.detach())
        real_imgs_aug_out = D(real_imgs_aug)


        # 1.GAN Generator loss
        if gan_type == "rgan" or gan_type == "ragan":
            GAN_G_loss = generator_loss(gan_type, fake_imgs_out[-2], real_imgs_out[-2])
        else:
            GAN_G_loss = generator_loss(gan_type, fake_imgs_out[-2])

        # 2.Feature Matching loss
        criterionFeat = torch.nn.L1Loss()
        GAN_Feature_Match_loss = Tensor(1).fill_(0)
        num_feature_maps = 4
        lambda_fm_weight = [1, 1, 1, 1]  # replace to attention module
        for i_map in range(num_feature_maps):
            fm_loss = criterionFeat(fake_imgs_out[i_map], real_imgs_out[i_map])
            fm_aug_loss = criterionFeat(fake_imgs_aug_out[i_map], real_imgs_aug_out[i_map])
            GAN_Feature_Match_loss += (fm_loss + fm_aug_loss) * lambda_fm_weight[i_map]

        # 3.MMD thriplet loss
        anc = fake_imgs_out[-1]
        pos = real_imgs_out[-1]
        neg = real_imgs_aug_out[-1]
        # criterionThrip = MMD_TripletLoss(margin=5)
        # criterionThrip = COSSIM_TripletLoss(margin=0.01)
        criterionTrip = HYPERSPHERE_TripletLoss(margin=config.margin)
        Triplet_loss = criterionTrip(anc, pos, neg) 

        # g_loss = 1. + 2. + 3.
        lambda_gan_g = config.lambda_gan_g
        lambda_gan_fm = 1.0
        lambda_trip = 10.0
        GAN_G_loss = GAN_G_loss * lambda_gan_g
        GAN_Feature_Match_loss = GAN_Feature_Match_loss * lambda_gan_fm
        GAN_Triplet_loss = Triplet_loss * lambda_trip

        g_loss = GAN_G_loss + GAN_Feature_Match_loss + GAN_Triplet_loss
        g_loss.backward()
        opt_G.step()

        # -----------------
        # train discriminator
        # -----------------
        opt_D.zero_grad()

        # Calculate Discriminator loss
        real_images_out = D(real_imgs)  # [fmap1, fmap2, fmap3, fmap4, final_out, project_out]
        fake_images_out = D(fake_imgs.detach())

        # 1.GAN Discriminator loss
        lambda_gan_d = 1.0
        lambda_gp = 1.0
        if gan_type == "wgan-gp" or gan_type == "wgan-lp":
            # calculate gp need to input `data`, or the imgs's grad will be calculated twice
            GP = lambda_gp * gradient_penalty(gan_type, D, real_imgs.data, fake_imgs.data)
        else:
            GP = 0

        d_loss = (discriminator_loss(gan_type, real_images_out[-2], fake_images_out[-2]) + GP) * lambda_gan_d
        d_loss.backward()
        opt_D.step()

        # -----------------
        # checkpoint
        # -----------------
        # save loss_log.txt to /work_dir/checkpoints_v3
        save_loss(checkpoint_path, epoch, n_epochs, i, len(dataloader), GAN_G_loss.item(), GAN_Feature_Match_loss.item(), GAN_Triplet_loss.item(), g_loss.item(), d_loss.item())

        # save samples to /work_dir/checkpoints
        save_samples(sample_path, sample_interval, epoch, i, len(dataloader), fake_imgs.data)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [GAN G loss: %f] [GAN Feature Match loss: %f] [GAN Triplet loss: %f] [G loss: %f] [D loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), GAN_G_loss, GAN_Feature_Match_loss, GAN_Triplet_loss, g_loss.item(), d_loss.item()), end='\n'
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
