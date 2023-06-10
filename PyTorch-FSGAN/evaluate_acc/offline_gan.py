# --------------------------
# generate samples
# --------------------------
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
import re
import shutil


def generator_samples(model_path, n=1500):
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
    if int(epoch) > 9000:
        generator_path = "../work_dir/generator_v4_10_num1500/" + gan_type + \
            "/" + data_path.split('/')[-1] + '/' + model_path.split('/')[-2] + '/epoch' + epoch  # generate according to epoch
        os.makedirs(generator_path, exist_ok=True)

        # generate images and save
        z_dim = checkpoint["z_dim"]
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n, z_dim))), requires_grad=False)
        G.eval()
        fake_imgs = G(z)
        for i, img in tqdm(enumerate(fake_imgs.cpu())):
            save_image(img, generator_path + "/%d.jpg" % i, normalize=True)


def generator_samples_epoch(gan_path):
    gan_root = Path(gan_path)
    for cls_ in tqdm(gan_root.iterdir()):  # Cr/...
        print(cls_)
        for type_ in cls_.iterdir():  # GAN_G1_FMLattn...
            for model in type_.glob("*G.pth"):  # ...net_G.pth
                # epoch = str(model).split('/')[-1].split('_')[0]
                model_path = str(model)
                generator_samples(model_path, n=1500)


def make_gan_all(type_="wgan-gp", lambda_gan_g=5e-2):
    source_root = "../work_dir/generator_v4_3080/" + type_
    target_root = "../dataset/NEU/NEU-50-r64-pad/" + type_ + '-my'
    for cls_ in os.listdir(source_root):
        target = target_root + '/' + cls_
        os.makedirs(target, exist_ok=True)
        source_cls = source_root + '/' + cls_
        expriment = list(filter(lambda x: re.search(str(lambda_gan_g), x) != None, os.listdir(source_cls)))[0]
        source = source_cls + '/' + expriment + '/epoch10000'
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(src=source, dst=target)


if __name__ == '__main__':
    generator_samples_epoch("../work_dir/checkpoints_v4_10/wgan-gp")
    # generator_samples_epoch("../work_dir/checkpoints_v4_10/wgan-gp")

    # offline gan
    # 1. type_ = wgan-gp
    # 2. lambda_gan_g = 5e-2
    # 3. epoch10000
    # train50/30/10
    # make_gan_all()
    pass