import numpy as np
import ot
import torch
from swd import read_images


def distance(X, Y, sqrt=False):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)
    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))
    del X, X2, Y, Y2
    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()
    return M


def wasserstein_distance(real, fake, k=1, sigma=1, sqrt=True):
    Mxy = distance(real, fake, False)
    if sqrt:
        Mxy = Mxy.abs().sqrt()
    emd = ot.emd2([], [], Mxy.numpy())
    return emd


if __name__ == '__main__':
    real_path = "../dataset/NEU/NEU-300-r64/Cr"
    fake_path = "../work_dir/generator/wgan-gp/Cr"
    real_images = read_images(real_path)
    fake_images = read_images(fake_path)

    print(wasserstein_distance(real_images, fake_images))
