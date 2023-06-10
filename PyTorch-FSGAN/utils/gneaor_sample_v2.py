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
    if int(epoch) > 9500:
        generator_path = "../work_dir/generator_v2_10/" + gan_type + \
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


if __name__ == '__main__':
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/dragan")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/gan")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/hinge")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/lsgan")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/ragan")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/wgan-gp")
    # generator_samples_epoch("../work_dir/checkpoints_v2_10/wgan-lp")
    generator_samples_epoch("../work_dir/checkpoints_v2_10/sphere")  # sphere GAN Generator is same.

    pass
