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
parser.add_argument('--batch_size', type=int, default=15)
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

dname = args.dataset_dir_S.replace('/','-')+'_'+args.dataset_dir_T.replace('/','-') +'_UNIT'
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

discriminatorS = models.Discriminator_UNIT().cuda()
discriminatorT = models.Discriminator_UNIT().cuda()
generator = models.Generator_UNIT(Z_dim).cuda()
encoder = models.Encoder_UNIT().cuda()

optim_discS = optim.Adam(filter(lambda p: p.requires_grad, discriminatorS.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_discT = optim.Adam(filter(lambda p: p.requires_grad, discriminatorT.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.9))
optim_enc = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.9))

iters = 0
loss_name = ['genS', 'discS', 'genT', 'discT']
loss_list = dict()
for loss in loss_name:
    loss_list[loss] = []

def train(epoch):
    for batch_idx, (data, TSrget) in enumerate(loader_T):

	num = data.size(0)
	dataPS, dataPT = data[:,:,:,:64].cuda(), data[:,:,:,64:].cuda()
        
        for _, (data2, _) in enumerate(loader_S):
	    dataTS = data2[:,:,:,:64].cuda()
            break

	muPS, logvarPS, zrPS = encoder(dataPS, domain='S')
        muPT, logvarPT, zrPT = encoder(dataPT, domain='T')
        muTS, logvarTS, zrTS = encoder(dataTS, domain='S')        

        reconPS, reconPS2B = generator(zrPS, domain='S'), generator(zrPS, domain='T')
        reconPT, reconPT2A = generator(zrPT, domain='T'), generator(zrPT, domain='S')
        reconTS, reconTS2B = generator(zrTS, domain='S'), generator(zrTS, domain='T')

        for i in range(disc_iters):
                z = Variable(torch.randn(num, Z_dim).cuda())
                randA, randB = generator(z, domain='S'), generator(z, domain='T')
                optim_discS.zero_grad()
                optim_discT.zero_grad()
                disc_lossS = nn.ReLU()(1.0 - discriminatorS(dataTS)).mean() + \
                                nn.ReLU()(1.0 + discriminatorS(randA.detach())).mean()

                disc_lossT = nn.ReLU()(1.0 - discriminatorT(dataPT)).mean() + \
                                nn.ReLU()(1.0 + discriminatorT(randB.detach())).mean()*0.5 + \
                                nn.ReLU()(1.0 + discriminatorT(reconTS2B.detach())).mean()*0.5

                disc_loss = (disc_lossS + disc_lossT)*0.5
                (disc_loss*0.1).backward()

                optim_discS.step()
                optim_discT.step()

        loss_list['discS'].append(disc_lossS.item()) 
        loss_list['discT'].append(disc_lossT.item()) 

        optim_gen.zero_grad()
        optim_enc.zero_grad()        
        randA, randB = generator(z, domain='S'), generator(z, domain='T')

        gen_lossS = -discriminatorS(randA).mean()
        gen_lossT = -discriminatorT(randB).mean()
        gen_loss = (gen_lossS + gen_lossT)*0.5

        recon_img_loss = torch.mean((dataPS-reconPS)**2) + torch.mean((dataPT-reconPT)**2) + torch.mean((dataTS-reconTS)**2) 
        recon_pair_loss = torch.mean((dataPT-reconPS2B)**2) + torch.mean((dataPS-reconPT2A)**2)

        muPS, logvarPS, zrPS = encoder(dataPS, domain='S')
        muPT, logvarPT, zrPT = encoder(dataPT, domain='T')
        muTS, logvarTS, zrTS = encoder(dataTS, domain='S')   
        muTS2B, logvarTS2B, zrTS2B = encoder(reconTS2B, domain='T')   

        KLD_TS = -0.5 * torch.mean(1 + logvarTS - muTS.pow(2) - logvarTS.exp())
        KLD_PT = -0.5 * torch.mean(1 + logvarPT - muPT.pow(2) - logvarPT.exp())        
        KLD_TS2B = -0.5 * torch.mean(1 + logvarTS2B - muTS2B.pow(2) - logvarTS2B.exp())
      
	img_loss = recon_img_loss + recon_pair_loss
	KLD = KLD_TS + KLD_PT + KLD_TS2B

	(img_loss + KLD*0.001+gen_loss*0.1).backward()
	optim_gen.step()
        optim_enc.step()

        loss_list['genS'].append(gen_lossS.item()) 
        loss_list['genT'].append(gen_lossT.item()) 

	global iters
        if iters % args.save_model_period == 0: # and iters % 500 == 0:
                torch.save(discriminatorS, os.path.join(checkpoint_dir, 'disc_S_{}'.format(epoch)))
                torch.save(discriminatorT, os.path.join(checkpoint_dir, 'disc_T_{}'.format(epoch)))
                torch.save(encoder, os.path.join(checkpoint_dir, 'enc_{}'.format(epoch)))
                torch.save(generator, os.path.join(checkpoint_dir, 'gen_{}'.format(epoch)))

        if iters % args.save_img_period == 0:
                out1, out2 = generator(z, domain='S'), generator(z, domain='T')

                save_img(dataPS.detach(), out_dir, 'dataPS', iters)
                save_img(reconPS.detach(), out_dir, 'reconPS', iters)
                save_img(dataPT.detach(), out_dir, 'dataPT', iters)
                save_img(reconPT.detach(), out_dir, 'reconPT', iters)
                save_img(dataTS.detach(), out_dir, 'dataTS', iters)
                save_img(reconTS.detach(), out_dir, 'reconTS', iters)
                save_img(out1.data.detach(), out_dir, 'out1', iters)
                save_img(out2.data.detach(), out_dir, 'out2', iters)

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
            s += ' recon_img_loss: {} recon_pair_loss: {}'.format(round(recon_img_loss.item(), 4), round(recon_pair_loss.item(), 4))
            print s
	iters += 1

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

for epoch in range(30000):
    train(epoch)
