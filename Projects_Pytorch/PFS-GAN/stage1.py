import warnings
warnings.filterwarnings("ignore")

import argparse
import scipy
import scipy.misc
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


dname = args.dataset_dir.replace('/','-')  + '_stage1'
checkpoint_dir = 'checkpoints_64x64_'+ dname  

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(args.dataset_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

Z_dim = 64
disc_iters = 5

discriminator = models.Discriminator().cuda()
generator = models.Generator(Z_dim*2).cuda()
encoder_c = models.Encoder(size_num=Z_dim).cuda()
encoder_t = models.Encoder(size_num=Z_dim).cuda()

v2 = models.vgg().cuda()

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.5,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.999))
optim_enc_c = optim.Adam([{'params':encoder_c.parameters()}], lr=args.lr, betas=(0.9,0.999))
optim_enc_t = optim.Adam([{'params':encoder_t.parameters()}], lr=args.lr, betas=(0.9,0.999))

iters = 0
loss_name = ['KLD_c', 'KLD_t']
loss_name += ['recon_img_loss', 'recon_t_loss', 'perceptual_loss']
loss_name += ['gen', 'disc']

loss_list = dict()
for loss in loss_name:
    loss_list[loss] = []
fg = False
z_fixed = torch.randn(64, 128).cuda()
def train(epoch):
    global loss_name, loss_list, iters, fg
    for batch_idx, (data, target) in enumerate(loader):
        optim_enc_c.zero_grad()
        optim_enc_t.zero_grad()
        optim_gen.zero_grad()
        v2.zero_grad()
        num = data.size(0)

        data, target = Variable(data.cuda()), Variable(target.cuda())
        data = data[:,:,:,:64]

        mu_c, logvar_c, zr_c = encoder_c(data)
        mu_t, logvar_t, zr_t = encoder_t(data)
        zr = torch.cat([zr_t, zr_c], 1)	

        if iters != 0 and fg: 
            for iiii in range(disc_iters):
                z = Variable(torch.randn(num, Z_dim*2).cuda()   )
                optim_disc.zero_grad()
                out_rand = generator(z)

                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(out_rand)).mean()*0.5 + nn.ReLU()(1.0 + discriminator(generator(zr).detach())).mean()*0.5
                (disc_loss*0.01).backward()
                optim_disc.step()

            optim_enc_t.zero_grad()
            optim_enc_c.zero_grad()
            optim_gen.zero_grad()
        
            gen_loss_v = -discriminator(generator(z)).mean()
            gen_loss_r = -discriminator(generator(zr.detach())).mean()
            gen_loss = (gen_loss_v + gen_loss_r)*0.5
            (gen_loss*0.01).backward()
            loss_list['gen'].append(gen_loss.item()); loss_list['disc'].append(disc_loss.item())


        # z = Variable(torch.randn(num, Z_dim*2).cuda())
        z1 = Variable(torch.randn(num, Z_dim).cuda())
        # out_rand = generator(z)

        mu_c, logvar_c, zr_c = encoder_c(data)
        mu_t, logvar_t, zr_t = encoder_t(data)
        zr = torch.cat([zr_t, zr_c], 1)
        zr2 = torch.cat([z1, zr_c], 1)
    #   mu_rand, logvar_rand, zr_rand = encoder_c(out_rand)

        recon_img = generator(zr) 
        recon_img2 = generator(zr2) 
        KLD_c = -0.5 * torch.mean(1 + logvar_c - mu_c.pow(2) - logvar_c.exp()) 
        KLD_t = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) 
        mu_t2, logvar_t2, zr_t2 = encoder_t(recon_img2)
    
        recon_img_loss = torch.mean((recon_img-data)**2) + torch.mean(torch.abs(recon_img-data))
        # print(recon_img_loss.item())
        f_recon_img, f_recon_img2, f_data = v2(recon_img), v2(recon_img2), v2(data)
        for i in range(3):
            recon_img_loss += torch.mean(torch.abs(f_recon_img[i]-f_data[i]))*0.01
    #       recon_img_loss += torch.mean((f_recon_img[i]-f_data[i])**2)*0.01
    #       print(torch.mean((f_recon_img[i]-f_data[i])**2).item())
        perceptual_loss = torch.mean((f_recon_img2[2]-f_data[2])**2)*0.01
        recon_t_loss = torch.mean((z1-zr_t2)**2)
        if recon_img_loss.item() < 0.1:
            fg = True

        recon_loss = recon_img_loss*1 + recon_t_loss*0.1 + perceptual_loss
        KLD_loss = KLD_c + KLD_t

        loss_list['KLD_c'].append(KLD_c.item())
        loss_list['KLD_t'].append(KLD_t.item())
        loss_list['recon_img_loss'].append(recon_img_loss.item()); 
        loss_list['recon_t_loss'].append(recon_t_loss.item()); 
        loss_list['perceptual_loss'].append(perceptual_loss.item())
        (recon_loss+KLD_loss*0.1).backward()
        optim_enc_c.step()
        optim_enc_t.step()
        optim_gen.step()


        if iters % 10 == 0:
            if iters % 500 == 0: 
                torch.save(discriminator, os.path.join(checkpoint_dir, 'disc_{}'.format(epoch)))
                torch.save(generator, os.path.join(checkpoint_dir, 'gen_{}'.format(epoch)))
                torch.save(encoder_t, os.path.join(checkpoint_dir, 'enc_t_{}'.format(epoch)))
                torch.save(encoder_c, os.path.join(checkpoint_dir, 'enc_c_{}'.format(epoch)))


            if iters % 100 == 0:
                out_rand = generator(z_fixed)
                save_img(recon_img.detach(), out_dir, 'recon_img', iters)
                save_img(recon_img2.detach(), out_dir, 'recon_img2', iters)
                save_img(data.detach(), out_dir, 'origin_img', iters)
#                    if iters != 0 and recon_img_loss.item()<0.05:
                save_img(out_rand.detach(), out_dir, 'origin_rand', iters)

                for loss in loss_name:
                    if len(loss_list[loss]) > 200:
                        plt.plot(np.arange(len(loss_list[loss][200:])), loss_list[loss][200:], label=loss)

                plt.legend()
                plt.savefig(os.path.join(out_dir, str(iters)+'_loss.png'))
                plt.clf()

            from time import gmtime, strftime
            s = strftime("%H:%M:%S", gmtime()) +  ' epoch: ' + str(epoch) + ' iters: ' + str(iters)
            if (iters > 10000 and iters % 10 == 0) or True: 
                if iters != 0 and fg:
                    s += ' disc: ' + str(round(disc_loss.item(), 4)) + ' gen: ' + str(round(gen_loss.item(), 4))         
                s += ' recon_img: ' + str(round(recon_img_loss.item(), 4)) + ' recon_t: '+str(round(recon_t_loss.item(), 4)) + ' perceptual: '+str(round(perceptual_loss.item(), 4))
                s += ' KLD_c: ' + str(round(KLD_c.item(), 4)) + ' KLD_t: ' + str(round(KLD_t.item(), 4)) 
                print(s)
        iters += 1

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

out_dir = 'out_' + dname
if os.path.isdir(out_dir) is False:
    os.makedirs(out_dir)
if os.path.isdir(checkpoint_dir) is False:
    os.makedirs(checkpoint_dir)

for epoch in range(30000):
    train(epoch)

