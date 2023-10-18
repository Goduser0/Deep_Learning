import warnings
warnings.filterwarnings("ignore")

import argparse
import scipy
import scipy.misc
from time import gmtime, strftime
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
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--dataset_dir_S', type=str, required=True)
parser.add_argument('--dataset_dir_T', type=str, required=True)
parser.add_argument('--gpu', type=str, required=True)

parser.add_argument('--save_model_period', type=int, default=500)
parser.add_argument('--save_img_period', type=int, default=50)
parser.add_argument('--plot_period', type=int, default=100)
parser.add_argument('--display_period', type=int, default=1)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dname = args.dataset_dir_S.replace('/','-')+'_'+args.dataset_dir_T.replace('/','-') +'_CoGAN'#+'_'+str(args.use_feat)+'_ratio:'+str(ratio)+'_model:'+args.model
checkpoint_dir = 'checkpoints_64x64_'+ dname
out_dir = 'out_' + dname

if os.path.isdir(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_S = datasets.ImageFolder(args.dataset_dir_S, transform=transform)
loader_S = torch.utils.data.DataLoader(dataset_S, batch_size=args.batch_size, shuffle=True, num_workers=2)
dataset_T = datasets.ImageFolder(args.dataset_dir_T, transform=transform)
loader_T = torch.utils.data.DataLoader(dataset_T, batch_size=args.batch_size, shuffle=True, num_workers=2)

Z_dim = 128
disc_iters = 1

discriminator = models.Discriminator_cogan().cuda()
generator = models.Generator_cogan(Z_dim).cuda()

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

iters = 0
loss_name = ['genS', 'discS', 'genT', 'discT']
loss_list = dict()
for loss in loss_name:
    loss_list[loss] = []

def train(epoch):
        for batch_idx, (data, target) in enumerate(loader_T):

                num = data.size(0)
                dataPS, dataPT = data[:,:,:,:64].cuda(), data[:,:,:,64:].cuda()
                
                for _, (data2, _) in enumerate(loader_S):
                        dataTA = data2[:,:,:,:64].cuda()
                        break

                for _ in range(disc_iters):
                        z = Variable(torch.randn(num, Z_dim).cuda())
                        randA, randB = generator(z, domain='S'), generator(z, domain='T')
                        optim_disc.zero_grad()
                        disc_lossS = nn.ReLU()(1.0 - discriminator(dataTA, domain='S')).mean() + \
                                        nn.ReLU()(1.0 + discriminator(randA.detach(), domain='S')).mean()

                        disc_lossT = nn.ReLU()(1.0 - discriminator(dataPT, domain='T')).mean() + \
                                        nn.ReLU()(1.0 + discriminator(randB.detach(), domain='T')).mean()

                        disc_loss = (disc_lossS + disc_lossT)*0.5
                        (disc_loss).backward()

                        optim_disc.step()

                loss_list['discS'].append(disc_lossS.item()) 
                loss_list['discT'].append(disc_lossT.item()) 

                optim_gen.zero_grad()    
                randA, randB = generator(z, domain='S'), generator(z, domain='T')

                gen_lossS = -discriminator(randA, domain='S').mean()
                gen_lossT = -discriminator(randB, domain='T').mean()
                gen_loss = (gen_lossS + gen_lossT)*0.5

                (gen_loss).backward()
                optim_gen.step()

                loss_list['genS'].append(gen_lossS.item()) 
                loss_list['genT'].append(gen_lossT.item()) 

                global iters
                if iters % args.save_model_period == 0: # and iters % 500 == 0:
                        torch.save(discriminator, os.path.join(checkpoint_dir, 'disc_{}'.format(epoch)))
                        torch.save(generator, os.path.join(checkpoint_dir, 'gen_{}'.format(epoch)))

                if iters % args.save_img_period == 0:
                        out1, out2 = generator(z, domain='S'), generator(z, domain='T')

                        save_img(out1.detach(), out_dir, 'rand_S', iters)
                        save_img(out2.detach(), out_dir, 'rand_T', iters)

                if iters % args.plot_period == 0:
                        for loss in loss_name:
                                plt.plot(np.arange(len(loss_list[loss])), loss_list[loss], label=loss)

                        plt.legend()
                        plt.savefig(os.path.join(out_dir, str(iters)+'_loss.png'))
                        plt.clf()

                if iters % args.display_period == 0:          
                        s = '{} epoch: {} iters: {}'.format(strftime("%H:%M:%S", gmtime()), epoch, iters) 
                        s += ' discS: {} genS: {}'.format(round(disc_lossS.item(), 4), round(gen_lossS.item(), 4))
                        s += ' discT: {} genT: {}'.format(round(disc_lossT.item(), 4), round(gen_lossT.item(), 4))
                        print s
                iters += 1

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

for epoch in range(30000):
    train(epoch)
