import argparse
import random
from time import gmtime, strftime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--gpu', type=int, required=True)

parser.add_argument('--save_model_period', type=int, default=500)
parser.add_argument('--save_img_period', type=int, default=50)
parser.add_argument('--plot_period', type=int, default=100)
parser.add_argument('--display_period', type=int, default=10)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dname = args.dataset_dir.replace('/','-')+'_baselineS' 
checkpoint_dir = 'checkpoints_64x64_'+ dname  
out_dir = 'out_' + dname

if os.path.isdir(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)

if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)

Z_dim = 128

### Datasets' Settings
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(args.dataset_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

discriminator = models.Discriminator_patch(channels=3).cuda()
generator = models.Generator(Z_dim).cuda()

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.5,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.9))

disc_iters = 1
iters = 0

loss_name = ['gen', 'disc']
loss_list = dict()
for loss in loss_name:
    loss_list[loss] = []

zc = Variable(torch.randn(50, Z_dim).cuda())

def train(epoch):
    global loss_name, loss_list, iters
    for batch_idx, (data, target) in enumerate(loader):

	optim_gen.zero_grad()
	num = data.size(0)

        data, target = Variable(data.cuda()), Variable(target.cuda())
        data = data[:,:,:,64:]

	for _ in range(disc_iters):
		z = Variable(torch.randn(num, Z_dim).cuda())
		optim_disc.zero_grad()
		out_rand = generator(z)
		real_img_loss, real_patch_loss = discriminator(data)
		fake_img_loss, fake_patch_loss = discriminator(out_rand)

		disc_patch_loss = nn.ReLU()(1.0 - real_patch_loss).mean() + nn.ReLU()(1.0 + fake_patch_loss).mean()
                disc_img_loss = nn.ReLU()(1.0 - real_img_loss).mean() + nn.ReLU()(1.0 + fake_img_loss).mean()

		disc_loss = (disc_patch_loss + disc_img_loss)*0.5
		(disc_loss).backward()
		optim_disc.step()
                loss_list['disc'].append(disc_loss.item())

	optim_gen.zero_grad()
	fake_img_loss, fake_patch_loss = discriminator(generator(z))
	gen_img_loss, gen_patch_loss = -fake_img_loss.mean() , -fake_patch_loss.mean()
	gen_loss = (gen_patch_loss + gen_img_loss)*0.5
	(gen_loss).backward()
	optim_gen.step()
        loss_list['gen'].append(gen_loss.item()) 

        
        if iters % args.save_model_period == 0: 
                torch.save(discriminator, os.path.join(checkpoint_dir, 'disc_{}'.format(epoch)))
                torch.save(generator, os.path.join(checkpoint_dir, 'gen_{}'.format(epoch)))

        if iters % args.save_img_period == 0: 
                out_rand = generator(zc.detach())
                save_img(data.detach(), out_dir, 'origin_img', iters)
                save_img(out_rand.detach(), out_dir, 'origin_rand', iters)

        if iters % args.plot_period == 0:
                for loss in loss_name:
                        plt.plot(np.arange(len(loss_list[loss])), loss_list[loss], label=loss)

                plt.legend()
                plt.savefig(os.path.join(out_dir, str(iters)+'_loss.png'))
                plt.clf()

        if iters % args.display_period == 0:	    
            s = '{} epoch: {} iters: {}'.format(strftime("%H:%M:%S", gmtime()), epoch, iters) 
	    s += ' disc: {} gen: {}'.format(round(disc_loss.item(), 4), round(gen_loss.item(), 4))
            print s

	iters += 1


for epoch in range(20000):
    train(epoch)
