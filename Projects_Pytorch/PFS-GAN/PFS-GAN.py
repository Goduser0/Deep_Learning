#coding=utf-8
import warnings
warnings.filterwarnings("ignore")

from skimage import color
import random
import argparse
import os
import scipy
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import save_img

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--train_dataset', type=str, required=True)
parser.add_argument('--test_dataset', type=str, required=True)
parser.add_argument('--source_dataset', type=str, required=True)
parser.add_argument('--recon_ratio', type=float, default=1.0)
parser.add_argument('--gan_ratio', type=float, default=1.0)
parser.add_argument('--relation_ratio', type=float, default=1.0)
parser.add_argument('--gpu_idx', type=str, default='1')
parser.add_argument('--model_name', type=str, required=True)

parser.add_argument('--save_model_period', type=int, default=500)
parser.add_argument('--save_img_period', type=int, default=50)
parser.add_argument('--plot_period', type=int, default=100)
parser.add_argument('--display_period', type=int, default=10)
args = parser.parse_args()

dname = args.train_dataset.replace('/','-')+'_gan:'+str(args.gan_ratio)+'_relation:'+str(args.relation_ratio)+'_recon:'+str(args.recon_ratio)+'_PFS-GAN'
checkpoint_dir = 'checkpoints_64x64_'+ dname  
out_dir = 'out_' + dname

if os.path.isdir(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)

if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)
		
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx 

### Datasets' Settings
train_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()]) 
test_transform =  transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(args.train_dataset, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)

test_dataset = datasets.ImageFolder(args.test_dataset, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

source_dataset = datasets.ImageFolder(args.source_dataset, transform=test_transform)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=50, shuffle=True, num_workers=2, drop_last=False)

### Models' Initialization
gen_name = 'gen_'+args.model_name 
enc_c_name = 'enc_c_' +args.model_name 
# Generator
t_generator, s_generator = torch.load(gen_name), torch.load(gen_name)
optim_gen  = optim.Adam(t_generator.parameters(), lr=args.lr, betas=(0.5, 0.9))

# Encoder
encoder_c, encoder_t = torch.load(enc_c_name), torch.load(enc_c_name) 
optim_enc_t = optim.Adam(encoder_t.parameters(), lr=args.lr, betas=(0.5, 0.9))

# Discriminator
if args.gan_ratio != 0:
    t_discriminator_patch = models.Discriminator_patch(channels=3).cuda()
    optim_disc_patch = optim.Adam(filter(lambda p: p.requires_grad, t_discriminator_patch.parameters()), lr=args.lr, betas=(0.0,0.9))

# Relation
if args.relation_ratio != 0:
	R = models.Relation().cuda()
	R.set(encoder_c)
	optim_R = optim.Adam(R.model_bn1_B.parameters(), lr=args.lr, betas=(0.5, 0.9))

# Feature Extractor    
F = models.vgg().cuda().eval()

### Lab shuffle
lab = []
def preprocessing(data):
	new_data = data.clone()
	global iters, lab
	# Record min/max a/b
	if iters == 0:
		max_a, min_a, max_b, min_b = -10000, 10000, -10000, 10000
		for b in range(new_data.size(0)):
			data2 = new_data[b].cpu().data.numpy().transpose(1,2,0).astype(float) #*255
			data2 = color.rgb2lab(data2)
			max_a, min_a = max(max_a, np.max(data2[:,:,1])),  min(min_a, np.min(data2[:,:,1]))
			max_b, min_b = max(max_b, np.max(data2[:,:,2])),  min(min_b, np.min(data2[:,:,2]))
		lab = [[min_a, max_a], [min_b, max_b]]
        count = iters
	# Shuffle
	for b in range(new_data.size(0)):
		data2 = new_data[b].cpu().data.numpy().transpose(1,2,0).astype(float) #*255
		data2 = color.rgb2lab(data2)
		max_a, max_b = np.max(data2[:,:,1]), np.max(data2[:,:,2])
		min_a, min_b = np.min(data2[:,:,1]), np.min(data2[:,:,2])

		rand_a = (np.random.choice(201)*0.01-1)*(lab[0][1]-lab[0][0])*0.5
		rand_b = (np.random.choice(201)*0.01-1)*(lab[1][1]-lab[1][0])*0.5

		data2[:,:,1] = np.clip(data2[:,:,1] + rand_a, lab[0][0], lab[0][1])
		data2[:,:,2] = np.clip(data2[:,:,2] + rand_b, lab[1][0], lab[1][1])
		data2 = color.lab2rgb(data2) 
		new_data[b] = (torch.Tensor(data2.transpose(2,0,1)).cuda()-0.5)*2

        return torch.clamp(new_data, -1.0, 1.0)

### Loss/Probs Record
loss, probs = dict(), dict()
loss_name = ['gen_patch', 'disc_patch', 'gen_patch_img', 'disc_patch_img', 'recon_img_loss', 'KLD_a', 'recon_a_loss']
loss_name += ['R_loss1', 'R_loss2', 'R_loss3', 'R_loss4']
probs_name = ['prob_real_patch', 'prob_fake_patch']
probs_name += ['prob_real_img', 'prob_fake_img']

for l in loss_name: loss[l] = []
for p in probs_name: probs[p] = []

iters = 0

z_fixed = torch.randn(64, 128).cuda()
zc = torch.randn(64, 64)
z_same_zs1, z_same_zs2 = torch.cat([torch.randn(1, 64).repeat(64,1), zc], 1).cuda(), torch.cat([torch.randn(1,64).repeat(64,1), zc], 1).cuda()

def train(epoch):
	global iters
	global loss, probs        
	for batch_idx, (data, target) in enumerate(train_loader):
		bsz1 = data.size(0)

		# Paired Training Image
		img_s1 = Variable(data[:,:,:,:64].cuda())*2-1.
		img_t = Variable(data[:,:,:,64:].cuda()) 
		# Keep Origin / Lab-Version
		img_t_orig = img_t.clone()*2-1 
		img_t_lab = preprocessing(img_t.clone())
		# Load More Content Features
		for _, (data2, _) in enumerate(source_loader):
			img_s2 = data2[:,:,:,:64].cuda()
			mu_s2_c, logvar_s2_c, z_s2_c = encoder_c(img_s2)
			bsz2 = img_s2.size(0)
			break

		mu_s1_c, logvar_s1_c, z_s1_c = encoder_c(img_s1)
		mu_a_lab, logvar_a_lab, z_a_lab = encoder_t(img_t_lab)
		mu_a_orig, logvar_a_orig, z_a_orig = encoder_t(img_t_orig)

		# Reconstruct Two Version of Training Dataset
		img_t_recon_lab = t_generator(torch.cat([z_a_lab, z_s1_c], 1))
		img_t_recon_orig = t_generator(torch.cat([z_a_orig, z_s1_c], 1))

		bsz_ratio = int(np.ceil(bsz2*1.0/bsz1))
		rand_z_a_lab = z_a_lab.repeat(bsz_ratio, 1)[:bsz2]
		rand_z_a_orig = z_a_orig.repeat(bsz_ratio, 1)[:bsz2]
		# Synthesize Image From Appearance/Content Features
		img_t_s2_lab = t_generator(torch.cat([rand_z_a_lab, z_s2_c], 1))
		img_t_s2_orig = t_generator(torch.cat([rand_z_a_orig, z_s2_c], 1))
		img_s_s2_rand = s_generator(torch.cat([torch.randn(bsz2, 64).cuda(), z_s2_c], 1))		

		gen_total_loss = 0
		if args.recon_ratio != 0: 
			# Reconstruct Two Versions of Training Dataset
			v_img_t, v_img_recon_s = F(img_t_orig), F(img_t_recon_orig)
			recon_img_loss0, recon_img_loss1 = torch.mean((img_t_recon_orig-img_t_orig)**2), torch.mean((v_img_recon_s[1]-v_img_t[1])**2)
			recon_img_loss_orig = (recon_img_loss0) + recon_img_loss1*0.1

			v_img_s2, v_img_recon_s2 = F(img_t_lab), F(img_t_recon_lab)
			recon_img_loss0, recon_img_loss1 = torch.mean((img_t_recon_lab-img_t_lab)**2), torch.mean((v_img_recon_s2[1]-v_img_s2[1])**2)
			recon_img_loss_lab = (recon_img_loss0) + recon_img_loss1*0.1

			recon_img_loss = recon_img_loss_orig * 0.1 + recon_img_loss_lab

			# Regularize Appearance Feature of Lab Version
			mu_s_lab_recon, logvar_s_lab_recon, z_s_lab_recon = encoder_t(img_t_recon_lab)

			KLD_a = -0.5 * torch.mean(1 + logvar_a_lab - mu_a_lab.pow(2) - logvar_a_lab.exp())
			recon_a_loss = torch.mean((z_a_lab-z_s_lab_recon)**2)

			recon_loss = recon_img_loss + recon_a_loss*0.1 + KLD_a*0.001 #*(recon_img_loss.item()<0.01) #+ recon_share_loss
			gen_total_loss += recon_loss * args.recon_ratio

		if args.gan_ratio != 0:
			for _ in range(5):
				optim_disc_patch.zero_grad()
				prob_real_patch, prob_real_img = t_discriminator_patch(img_t_recon_orig.detach())
				prob_fake_patch, prob_fake_img = t_discriminator_patch(img_t_s2_orig.detach()) 
				
				disc_loss_patch = nn.ReLU()(1.0 - prob_real_patch).mean()*1 + nn.ReLU()(1.0 + prob_fake_patch).mean()
				disc_loss_img = nn.ReLU()(1.0 - prob_fake_img).mean() + nn.ReLU()(1.0 + prob_fake_img).mean()

				((disc_loss_patch+disc_loss_img*0.01)*args.gan_ratio).backward()
				optim_disc_patch.step()

				def extract(data):
					return torch.mean(nn.Sigmoid()(data)).item()

				probs['prob_real_patch'].append(extract(prob_real_patch)); 
				probs['prob_fake_patch'].append(extract(prob_fake_patch));

			prob_fake_patch, prob_fake_img = t_discriminator_patch(img_t_s2_orig)

			gen_loss_patch = -prob_fake_patch.mean()
			gen_loss_img = -prob_fake_img.mean()

			gen_total_loss += (gen_loss_patch+gen_loss_img*0.01)* args.gan_ratio 

		if args.relation_ratio != 0:
			# Training Relation Network
			optim_R.zero_grad()
			out_pair = R(img_s1, img_t_lab.detach())
			R_loss1 = torch.mean(torch.abs(out_pair-0))

			out_rand = R(img_s_s2_rand.detach(), img_t_s2_lab[0:1].repeat(bsz2, 1, 1, 1).detach())
			rand_dist = (mu_s2_c-mu_s1_c[0:1].repeat(bsz2, 1))**2 			
			R_loss2 = torch.mean(torch.abs(out_rand-rand_dist.detach()))
			(R_loss1 + R_loss2).backward()
			optim_R.step()
			optim_R.zero_grad()

			# Use Relation Network to Regularize Network
			out_pair = R(img_s_s2_rand.detach(), img_t_s2_lab) 
			R_loss3 = torch.mean(torch.abs(out_pair-0))
			out_rand = R(img_s1, img_t_s2_lab[0:1].repeat(bsz1, 1, 1, 1))
			rand_dist = (mu_s1_c-mu_s2_c[0:1].repeat(bsz1, 1))**2 
			R_loss4 = torch.mean(torch.abs(out_rand-rand_dist))

			R_loss = R_loss3 + R_loss4
			gen_total_loss += R_loss*args.relation_ratio 

		(gen_total_loss).backward()
		optim_gen.step()
		optim_enc_t.step()
		if args.relation_ratio != 0:
			optim_R.step()
			optim_R.zero_grad()

		optim_gen.zero_grad()
		optim_enc_t.zero_grad()  	

		if args.gan_ratio != 0 and iters % 10 == 0:

			loss['gen_patch'].append(round(gen_loss_patch.item()*args.gan_ratio, 3)); 
			loss['disc_patch'].append(round(disc_loss_patch.item()*args.gan_ratio, 3));
        
		def getValue(loss): return round(loss.item(), 3)

		if args.recon_ratio != 0 and iters % 10 == 0:
			loss['recon_img_loss'].append(getValue(recon_img_loss*args.recon_ratio)); 
			loss['recon_a_loss'].append(getValue(recon_a_loss*args.recon_ratio*0.1));
			loss['KLD_a'].append(getValue(KLD_a*args.recon_ratio*0.01));

		if args.relation_ratio != 0 and iters % 10 == 0:
			loss['R_loss1'].append(getValue(R_loss1*args.relation_ratio));
			loss['R_loss2'].append(getValue(R_loss2*args.relation_ratio));
			loss['R_loss3'].append(getValue(R_loss3*args.relation_ratio));
			loss['R_loss4'].append(getValue(R_loss4*args.relation_ratio));

		if iters % args.save_model_period == 0: 
			optim_gen.zero_grad()
			optim_enc_t.zero_grad()
			torch.save(t_generator, os.path.join(checkpoint_dir, 'gen_{}'.format(iters)))
			torch.save(encoder_t, os.path.join(checkpoint_dir, 'enc_t_{}'.format(iters)))

		if iters % args.save_img_period == 0: 
			img_rand_t, img_rand_s = s_generator(z_fixed), t_generator(z_fixed)			
			img_zs1, img_zs2 = t_generator(z_same_zs1), t_generator(z_same_zs2)

			if os.path.isdir(os.path.join(out_dir, 'train')) is False: os.makedirs(os.path.join(out_dir, 'train'))
			if os.path.isdir(os.path.join(out_dir, 'zs')) is False: os.makedirs(os.path.join(out_dir, 'zs'))
	
			save_img(img_t_s2_orig.detach(), os.path.join(out_dir, 'train'), 'st_orig', iters)
			save_img(img_t_lab.detach(), os.path.join(out_dir, "train"), 's_lab', iters)
			save_img(img_t_recon_lab.detach(), os.path.join(out_dir, 'train'), 's_lab_recon', iters)			
			save_img(img_zs2.detach(), os.path.join(out_dir, 'zs'), 'same_zs2', iters)
			save_img(img_zs1.detach(), os.path.join(out_dir, 'zs'), 'same_zs1', iters)

		if iters % args.plot_period == 0:
			for name in loss_name:
				if len(loss[name]) != 0: plt.plot(np.arange(len(loss[name])), loss[name], label=name)
			plt.legend()
			plt.savefig(os.path.join(out_dir, str(iters)+'_loss.png'))
			plt.clf()

			for name in probs_name:
				if len(probs[name]) != 0: plt.plot(np.arange(len(probs[name])), probs[name], label=name)

			plt.legend()
			plt.savefig(os.path.join(out_dir, str(iters)+'_prob.png'))
			plt.clf()

		if iters % args.display_period == 0:
			s = strftime("%H:%M:%S", gmtime())+' epoch '+str(epoch)+' count '+str(iters) + ' '
			for name in loss_name:
				if len(loss[name]) != 0:    
					s += '%s: %.3f ' % (name, loss[name][-1])
			print s
			              
		iters += 1
		torch.cuda.empty_cache()

for epoch in range(50000):
    if iters > 8000:
		break
    train(epoch)

