import torch
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))  # make sure correct Relative import third_party 
import cv2
import numpy as np
from third_party.GANMetric.swd_pytorch.swd import swd


def read_images(images_path):
    """ real all images in images_path
        return Torch.Tensor as [N, C, H, W] which normalize in [0, 1]."""
    images = []
    for i, img in enumerate(os.listdir(images_path)):
        img_path = images_path + '/' + img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.  # ToTensor
        # images[i] = img
        images.append(img)
    images = np.array(images)
    print(images.shape)
    return torch.FloatTensor(images)


def slice_wasserstein_distance(real_imgs, fake_imgs):
    """calculate Sliced Wasserstein Distance (SWD) according to images."""
    torch.manual_seed(42)
    # real_imgs = read_images(real_imgs)
    # fake_imgs = read_images(fake_imgs)
    out = swd(real_imgs, fake_imgs, device="cuda", n_repeat_projection=4, proj_per_repeat=128)
    return out


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def sliced_wasserstein_distance(real_images, fake_images, num_projections=1000,p=2,device='cuda'):
    # input to cuda, Float64 will be Float32, so First Reshape ! ! !
    # v2, is not input a image, rather than a feature samples. so this may have some problems.
    batchs = real_images.shape[0]
    
    # calculate SWD according to the original images.Rather than Features.
    real_images = real_images.reshape(batchs, -1).cuda()
    fake_images = fake_images.reshape(batchs, -1).cuda()

    dim = fake_images.size(1)
    projections = rand_projections(dim, num_projections).to(device)

    first_projections = real_images.matmul(projections.transpose(0, 1))
    second_projections = (fake_images.matmul(projections.transpose(0, 1)))

    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)


if __name__ == '__main__':
    # test 300 images calculate slice_wasserstein_distance
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    real_path = "../dataset/NEU/NEU-50-r64/train/Cr"
    fake_path = "../work_dir/generator/wgan-gp/Cr/epoch10000"
    real_images = read_images(real_path)
    fake_images = read_images(fake_path)

    print(slice_wasserstein_distance(real_images, fake_images))
    print(sliced_wasserstein_distance(real_images, fake_images))
