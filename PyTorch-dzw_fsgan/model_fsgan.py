from unittest import result
import torch
import numpy as np
import torch.nn as nn
from attention import SelfAttention as SelfAttn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary


class Generator(nn.Module):
    """ Generator 
    Conv2D : 
        H' = [(H-K+2p)/s]+1, Conv2D(3,8,3,stride=2,padding=1)(1, 3, 5, 5)=(1,8,3,3)
    ConvTranspose2d : (in_channels, out_channels, kernel_size, stride=1, padding=0,out_padding=0)
        padding with `0` in&out input feature map.
        H' = [H + (s - 1)*(H - 1) - k + 2(k - p - 1) / 1] + 1
        Simplify as H' = 2H + k - 2p - 2 when s = 2.
        Simplify as H' = H + k - 2p - 1  when s = 1.
    """

    def __init__(self, image_size=64, z_dim=128, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        # layer5 = []
        last   = []

        repeat_num = int(np.log2(self.imsize)) - 3
        multi = 2 ** repeat_num  # 8

        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * multi, 4)))  # ->(64x8,4,4)
        layer1.append(nn.BatchNorm2d(conv_dim * multi))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * multi  # layer1 output dim 64x8

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))  # ->(64x4,8,8)
        layer2.append(nn.BatchNorm2d(curr_dim // 2))
        layer2.append(nn.ReLU())

        curr_dim = curr_dim // 2  # layer2 output dim 64x4

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))  # ->(64x2,16,16)
        layer3.append(nn.BatchNorm2d(curr_dim // 2))
        layer3.append(nn.ReLU())

        curr_dim = curr_dim // 2  # layer3 output dim 64x2
        attn1_dim = curr_dim  # attn1 dim 64x2

        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))  # ->(64x1,32,32)
        layer4.append(nn.BatchNorm2d(curr_dim // 2))
        layer4.append(nn.ReLU())

        curr_dim = curr_dim // 2  # layer4 output dim 64x1
        attn2_dim = curr_dim  # attn2 dim 64x1

        # layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))  # ->(64x1,64,64)
        # layer5.append(nn.BatchNorm2d(curr_dim // 2))
        # layer5.append(nn.ReLU())

        # curr_dim = curr_dim // 2  # layer5 output dim 64x1
        # attn3_dim = curr_dim  # attn3 dim 64x1

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))  # ->(3,64,64)
        last.append(nn.Tanh())  # (-1,1)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        # self.l5 = nn.Sequential(*layer5)
        self.last = nn.Sequential(*last)

        self.attn1 = SelfAttn(attn1_dim)
        self.attn2 = SelfAttn(attn2_dim)
        # self.attn3 = SelfAttn(attn3_dim)

    def forward(self, z):
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)  # (B, z_dim, 1, 1)
        out = self.l1(z)  # (B, 64x8, 4, 4) -> BN -> ReLU
        out = self.l2(out)  # (B, 64x4, 8, 8) -> BN -> ReLU
        out = self.l3(out)  # (B, 64x2, 16, 16) -> BN -> ReLU
        out, p1 = self.attn1(out)  # attn1=128
        out = self.l4(out)  # (B, 64x1, 32, 32) -> BN -> ReLU
        out, p2 = self.attn2(out)  # attn2=64
        out = self.last(out)  # (B, 3, 64, 64) -> Tanh (-1, 1)

        return out


class Discriminator(nn.Module):
    """Discriminator
    All Conv2D use SpectralNorm, not use BatchNorm.
    Conv2D : (in_channels, out_channels, kernel_size, stride=1, padding=0)
        H' = [(H - k + 2p) / s] + 1
        Simplify as H' = [H / 2] when k=4, s=2, p=1
        Simplify as H' =  H - 3  when k=4, s=1, p=0
    """

    def __init__(self, image_size=64, conv_dim=64, gan_type=None):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))  # (1, 3, 64, 64) -> (1, 64, 32, 32)
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim  # layer1 output dim 64

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))  # -> (1, 128, 16, 16)
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2  # layer2 output dim 128

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))  # -> (1, 256, 8, 8)
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2  # layer3 output dim 256
        attn1_dim = curr_dim  # attn1 dim 256

        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))  # -> (1, 512, 4, 4)
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2  # layer4 output dim 512
        attn2_dim = curr_dim  # attn2 dim 512

        last.append(nn.Conv2d(curr_dim, 1, 4))  # -> (1, 1, 1, 1)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)

        self.attn1 = SelfAttn(attn1_dim)
        self.attn2 = SelfAttn(attn2_dim)

        self.block = nn.Linear(1, 512)


    def forward(self, x):
        results = []
        out = self.l1(x)  # (B, 64, 32, 32) -> LeakyReLU
        results.append(out)  # Feature map layer1

        out = self.l2(out)  # (B, 128, 16, 16) -> LeakyReLU
        results.append(out)  # Feature map layer2

        out = self.l3(out)  # (B, 256, 8, 8) -> LeakyReLU
        out, p1 = self.attn1(out)  # attn1=256
        results.append(out)  # Feature map layer3

        out = self.l4(out)  # (B, 512, 4, 4) -> LeakyReLU
        out, p2 = self.attn2(out)  # attn2=512
        results.append(out)  # Feature map layer4

        out = self.last(out)  # (B, 1, 1, 1)
        out = out.reshape(out.shape[0], -1)  # (B, 1)
        results.append(out)  # Final output to calculate GAN Loss

        out = self.block(out)  # (B, 512)
        results.append(out)  # Project output to calculate MMD Loss

        return results


if __name__ == '__main__':
    # test Generator
    G = Generator()
    z = torch.normal(0, 1, size=(32, 128))
    fake = G(z)
    print(z.shape, fake.shape)
    summary(G, z.shape, device="cpu")

    # test Discriminator
    D = Discriminator()
    pred = D(fake)
    # print(fake.shape, pred.shape, pred)
    print(len(pred))
    for i in range(len(pred)):
        print(pred[i].shape)
    summary(D, fake.shape, device="cpu")

    # # test SpectralNorm
    # conv = nn.Conv2d(1, 3, 4)
    # conv.weight = nn.Parameter(torch.randn(3, 1, 4, 4))
    # conv_w = conv.weight[0][0].reshape(4, 4)
    # print(conv_w.shape, torch.linalg.norm(conv_w, 2))
    # snm = SpectralNorm(conv)
    # snm_w  = snm.weight[0][0].reshape(4, 4)
    # print(snm_w.shape, torch.linalg.norm(snm_w, 2))