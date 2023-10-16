import models
import os
import numpy as np
import shutil
import scipy
import scipy.misc
import argparse

import matplotlib
matplotlib.use('Agg')
import  matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

from fid_score import * 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--target_dataset', type=str, required=True)
parser.add_argument('--source_dataset', type=str, required=True)
parser.add_argument('--base_dataset', type=str, required=True)
parser.add_argument('--target_folder', type=str, required=True)
parser.add_argument('--model_num', type=int, required=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

test_transform =  transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

source_dataset = datasets.ImageFolder(args.source_dataset, transform=test_transform)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=15, shuffle=True, num_workers=2, drop_last=False)

target_dataset = datasets.ImageFolder(args.target_dataset, transform=test_transform)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=15, shuffle=True, num_workers=2, drop_last=False)

enc_c = torch.load('enc_c_'+str(args.model_num))

def save_img(img, dir, name):
        n = int(np.sqrt(img.shape[0]))
        out_img = np.zeros((64*n,64*n,3))
        for k1 in range(n):
            for k2 in range(n):
                out_img[k1*64:(k1+1)*64,k2*64:(k2+1)*64,:] = img[k1*n+k2]
                scipy.misc.imsave(os.path.join(dir, name+'.jpg'), out_img)
        return out_img

def main():
	m_train, s_train = fid_score_by_folder(args.base_dataset)
	for i,j,k in os.walk(args.target_folder):
	    if 'checkpoint' not in i: continue
            f = open(i.replace('/','_')+'.txt', 'w')
	    fid_train_rand, fid_train_syn = [], []
	    for g in range(0, 10001, 500):
	        if os.path.isfile(os.path.join(i,'gen_'+str(g))) is False: continue

		if os.path.isdir('fid_out_rand') is True: shutil.rmtree('fid_out_rand'); 
		os.makedirs('fid_out_rand')
		if os.path.isdir('fid_out_syn') is True: shutil.rmtree('fid_out_syn'); 
                os.makedirs('fid_out_syn')

		gen = torch.load(os.path.join(i,'gen_'+str(g))).cuda()
		### Total Random Latent Code
		for t1 in range(100):
		    z = Variable(torch.randn(100, 128).cuda()) 
		    out_rand = gen(z).data.cpu().detach().numpy().transpose(0,2,3,1)*0.5+0.5
		    for t2 in range(100):
			scipy.misc.imsave('fid_out_rand/'+str(t1*100+t2)+'.jpg', np.clip(out_rand[t2], 0, 1))
		# Compute FID
		m_rand, s_rand = fid_score_by_folder('fid_out_rand')
		fid_rand_value = compute_fid(m_train, s_train, m_rand, s_rand)
		print('fid_rand:', fid_rand_value)
                fid_train_rand.append(fid_rand_value)
                
                if 'origin' in i or 'transfer' in i: continue
		### Synthesize Appearance/Content Latent Code
		# Extract Content Code
		zcs = None
		while True:
		    for batch_idx, (data, target) in enumerate(source_loader):
		        data = data[:,:,:,:64].cuda()
			_,_,zc = enc_c(data)
			if zcs is None:
                            zcs = zc.data.cpu().detach().numpy()
			else:
			    zcs = np.concatenate([zcs, zc.data.cpu().detach().numpy()], 0)
		    if zcs.shape[0] >= 10000:
			break
		zcs = zcs[:10000]

		# Mix Up The Appearance Code
		enc_t = torch.load(os.path.join(i, 'enc_t_'+str(g))).cuda()

		c = 0
	        cc = 0
		while True:
		    fg = False
		    for batch_idx, (data, target) in enumerate(target_loader):
		        data = data[:,:,:,64:].cuda()
                        _,_,zt = enc_t(data)
                        zc = torch.FloatTensor(zcs[c:c+zt.size(0)]).cuda()
			if len(zc) == 0:
			    fg = True
			    break
			z = torch.cat([zt, zc], 1)
			c += z.size(0)
			out_rand_tc = gen(z).cpu().detach().numpy().transpose(0,2,3,1)*0.5+0.5
                        for t in range(out_rand_tc.shape[0]):
                            scipy.misc.imsave('fid_out_syn/'+str(cc)+'.jpg', np.clip(out_rand_tc[t], 0, 1))
                            cc += 1
		    if fg == True:
			break
		# Compute FID
		m_syn, s_syn = fid_score_by_folder('fid_out_syn')
		fid_syn_value = compute_fid(m_train, s_train, m_syn, s_syn)
		print('fid_syn:', fid_syn_value)
                fid_train_syn.append(fid_syn_value)
		print("----------------------------------------")

	    f.write(i+' '+' '.join([str(s) for s in fid_train_rand])+'\n')
            f.write(i+' '+' '.join([str(s) for s in fid_train_syn])+'\n')
            f.close()	  

if __name__ == "__main__":
	main() 
