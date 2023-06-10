# --------------------------------------------------
# generate all images according to saved G.pths
# --------------------------------------------------
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))  # add sys path for relative import 
from model import Generator  # Generator is same between sphere and others.
from torchvision.utils import save_image
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import argparse
from pathlib import Path
from checkpoint import plot_loss


def generator_random_samples(n=50):
    """generate random samples from init Generator."""
    generator_path = "../work_dir/samples/random_samples/Sc"
    # dont load checkpoint
    G = Generator()
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n, 128))), requires_grad=False)
    G.eval()
    fake_imgs = G(z)
    for i, img in enumerate(fake_imgs):
        save_image(img, generator_path + "/%d.jpg" % i, normalize=True)


def generator_samples(model_path, n=50):
    """load model and checkpoint to generate fake images"""
    # torch.load checkpoint
    checkpoint = torch.load(model_path)

    # Generator
    G = Generator()
    G.load_state_dict(checkpoint["model_state_dict"])

    # save path 
    gan_type = checkpoint["gan_type"]
    data_path = checkpoint["dataset"]
    epoch = model_path.split('/')[-1].split('_')[0]
    generator_path = "../work_dir/generator/" + gan_type + \
        "/" + data_path.split('/')[-1] + '/epoch' + epoch  # generate according to epoch
    os.makedirs(generator_path, exist_ok=True)

    # generate images and save
    z_dim = checkpoint["z_dim"]
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n, z_dim))), requires_grad=False)
    G.eval()
    fake_imgs = G(z)
    for i, img in tqdm(enumerate(fake_imgs)):
        save_image(img, generator_path + "/%d.jpg" % i, normalize=True)


def generator_samples_epoch(gan_path):
    gan_root = Path(gan_path)
    for cls_ in tqdm(gan_root.iterdir()):  # Cr/...
        print(cls_)
        for t_dir in cls_.iterdir():  # 2022...dir
            for model in t_dir.glob("*G.pth"):  # ...net_G.pth
                # epoch = str(model).split('/')[-1].split('_')[0]
                model_path = str(model)
                generator_samples(model_path, n=50)


def plot_loss_all(gan_path):
    gan_root = Path(gan_path)
    for cls_ in gan_root.iterdir():
        for t_dir in cls_.iterdir():
            for log in t_dir.glob("*.txt"):
                plot_loss(str(log), mode="epoch")
                plot_loss(str(log), mode="batch")


if __name__ == '__main__':
    # generator_samples_epoch("../work_dir/checkpoints_v2/dragan")
    # generator_samples_epoch("../work_dir/checkpoints_v2/gan")
    # generator_samples_epoch("../work_dir/checkpoints_v2/hinge")
    # generator_samples_epoch("../work_dir/checkpoints_v2/lsgan")
    # generator_samples_epoch("../work_dir/checkpoints_v2/ragan")
    # generator_samples_epoch("../work_dir/checkpoints_v2/rgan")
    # generator_samples_epoch("../work_dir/checkpoints_v2/wgan-gp")
    # generator_samples_epoch("../work_dir/checkpoints_v2/wgan-lp")
    generator_samples_epoch("../work_dir/checkpoints_v2/sphere")  # sphere GAN Generator is same.

    # plot_loss_all("../work_dir/checkpoints_v2/dragan")
    # plot_loss_all("../work_dir/checkpoints_v2/gan")
    # plot_loss_all("../work_dir/checkpoints_v2/hinge")
    # plot_loss_all("../work_dir/checkpoints_v2/lsgan")
    # plot_loss_all("../work_dir/checkpoints_v2/ragan")
    # plot_loss_all("../work_dir/checkpoints_v2/rgan")
    # plot_loss_all("../work_dir/checkpoints_v2/wgan-gp")
    # plot_loss_all("../work_dir/checkpoints_v2/wgan-lp")
    # plot_loss_all("../work_dir/checkpoints_v2/sphere")
    
    # generator_random_samples()
    pass
