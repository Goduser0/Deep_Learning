import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torch import from_numpy
from torch import autograd

from numpy import array
import imgaug.augmenters as iaa

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import img_1to255, img_255to1

########################################################################################################
# CLASS: SelfAttention()
########################################################################################################
class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation="relu", k=8):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W*H)
        qk = torch.bmm(proj_query, proj_key)
        attention = self.softmax(qk)
        proj_value = self.value(x).view(B, -1, W*H)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out, attention
        

########################################################################################################
# CLASS: ConGAN_Generator()
########################################################################################################
class ConGAN_Generator(nn.Module):
    """
    forward:
        input: [batch_size, 128]
        output:[batch_size, 3, 64, 64]
    """
    def __init__(self, img_size=64, z_dim=128, conv_dim=64):
        super(ConGAN_Generator, self).__init__()
        
        self.img_size = img_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        repeat_num = int(np.log2(self.img_size)) - 3
        multi = 2 ** repeat_num
        
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim*multi, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim*multi))
        layer1.append(nn.ReLU())
        
        curr_dim = conv_dim * multi
        
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim // 2))
        layer2.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(curr_dim // 2))
        layer3.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(curr_dim // 2))
        layer4.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn2_dim = curr_dim
        
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
        
    def forward(self, z):
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        return out
        
########################################################################################################
# CLASS: ConGAN_Discriminator()
########################################################################################################
class ConGAN_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, 64, 64] 
        output:
        [
            [batch_size, 64, 32, 32],
            [batch_size, 128, 16, 16],
            [batch_size, 256, 8, 8],
            [batch_size, 512, 4, 4],
            [batch_size, 1],
            [batch_size, 64],
        ]
    """
    def __init__(self, img_size=64, conv_dim=64):
        super(ConGAN_Discriminator, self).__init__()
        self.img_size = img_size
        layer1 = [] 
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attn2_dim = curr_dim
        
        last.append(nn.Conv2d(curr_dim, 1, 4))
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
        
        self.MLP = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        
        self.u1 = nn.ConvTranspose2d(512, 512 // 2, 4, 2, 1)
        self.u2 = nn.ConvTranspose2d(256, 256 // 2, 4, 2, 1)
        self.u3 = nn.ConvTranspose2d(128, 128 // 2, 4, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.w4 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
                
    def forward(self, x):
        results = []
        out = self.l1(x)
        feature1 = out
        results.append(out)

        out = self.l2(out)
        feature2 = out
        results.append(out)
        
        out = self.l3(out)
        out, p1 = self.attn1(out)
        feature3 = out
        results.append(out)
        
        out = self.l4(out)
        out, p2 = self.attn2(out)
        feature4 = out
        results.append(out)
        
        out = self.last(out)
        out = out.reshape(out.shape[0], -1)
        results.append(out)
        
        out = self.MLP(out)
        results.append(out)
        
        feature4_up = self.u1(feature4)
        feature3 = feature3 + feature4_up
        feature3_up = self.u2(feature3)
        feature2 = feature2 + feature3_up
        feature2_up = self.u3(feature2)
        feature1 = feature1 + feature2_up
        feat_avg = self.avg(feature1).reshape(feature1.shape[0], -1)
        feat_attn = self.w4(feat_avg)
        feat_attn = self.softmax(feat_attn)
        
        for i in range(feat_attn.shape[1]):
            attn = feat_attn[:, i].reshape(feat_attn.shape[0], 1, 1, 1)
            results[i] = attn * results[i]

        return results

########################################################################################################
# FUNCTION: Cal_Gradient_Penalty()
########################################################################################################
def Cal_Gradient_Penalty(discriminator, real_imgs, fake_imgs, device):
    batch_size = real_imgs.shape[0]
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).to(device).expand_as(real_imgs)
    
    interpolated = eta*real_imgs + (1-eta)*fake_imgs.requires_grad_(True)
    prob_interpolated = discriminator(interpolated)[-2]
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.shape).to(device),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.reshape(gradients.shape[0], -1)
    GP = torch.mean(torch.square((gradients.norm(2, dim=1))))
    return GP
    
########################################################################################################
# CLASS: HYPERSPHERE_TripletLoss()
########################################################################################################
class HYPERSPHERE_TripletLoss(nn.Module):
    def __init__(self, margin):
        super(HYPERSPHERE_TripletLoss, self).__init__()
        self.margin = margin

    def project_to_hypersphere(self, p):
        p_norm = torch.norm(p, dim=1, keepdim=True) ** 2
        x = 2 * p / (p_norm + 1)
        y = (p_norm - 1) / (p_norm + 1)
        return torch.cat([x, y], dim=1)

    def hypersphere_distance(self, x1, x2, moment=5, eps=1e-6):
        p = self.project_to_hypersphere(x1)
        q = self.project_to_hypersphere(x2)
        p_norm = torch.norm(p, dim=1) ** 2
        q_norm = torch.norm(q, dim=1) ** 2
        top = p_norm * q_norm - p_norm - q_norm + torch.sum(4 * p * q, dim=1) + 1
        bottom = (p_norm + 1) * (q_norm + 1)
        sphere_d = torch.acos((top / bottom).clamp_(-1.0+eps, 1.0-eps))
        distance = 0
        for i in range(1, moment + 1):
            distance += torch.pow(sphere_d, i)
        return distance

    def forward(self, anchor, pos, neg):
        Loss = nn.TripletMarginWithDistanceLoss(distance_function=self.hypersphere_distance, margin=self.margin)
        return Loss(anchor, pos, neg)

########################################################################################################
# CLASS: ImgAugmentation()
########################################################################################################
class ImgAugmentation():
    def __init__(self):
        self.seed = 42
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=self.seed)
        self.aug_real = iaa.Sequential([
            iaa.SomeOf((1, 5),
                       [
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           self.sometimes(iaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                               iaa.Fliplr(random_state=self.seed),
                               iaa.Flipud(random_state=self.seed)
                           ], random_state=self.seed),
                           iaa.Crop(keep_size=True, random_state=self.seed)
                       ], random_order=True, random_state=self.seed)
        ], random_order=True, random_state=self.seed)
        self.aug_fake = iaa.Sequential([
            iaa.SomeOf((1, 5),
                       [
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           self.sometimes(iaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                               iaa.Fliplr(random_state=self.seed),
                               iaa.Flipud(random_state=self.seed)
                           ], random_state=self.seed),
                           iaa.Crop(keep_size=True, random_state=self.seed)
                       ], random_order=True, random_state=self.seed)
        ], random_order=True, random_state=self.seed)

    def imgRealAugment(self, imgs):
        device = imgs.device
        imgs = img_1to255(imgs)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = array(imgs.cpu())
        imgs_aug = self.aug_real.augment_images(imgs)
        imgs_aug = imgs_aug.transpose(0, 3, 1, 2)
        imgs_aug = img_255to1(imgs_aug)
        imgs_aug = from_numpy(imgs_aug)
        imgs_aug = imgs_aug.to(device)
        return imgs_aug

    def imgFakeAugment(self, imgs):
        device = imgs.device
        imgs = img_1to255(imgs)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = array(imgs.cpu())
        imgs_aug = self.aug_fake.augment_images(imgs)
        imgs_aug = imgs_aug.transpose(0, 3, 1, 2)
        imgs_aug = img_255to1(imgs_aug)
        imgs_aug = from_numpy(imgs_aug)
        imgs_aug = imgs_aug.to(device)
        return imgs_aug


if __name__ == "__main__":
    z = torch.randn(16, 128)
    ConGAN_G = ConGAN_Generator()
    # print(ConGAN_G.forward(z).shape)
    
    img = torch.randn(16, 3, 64, 64)
    ConGAN_D = ConGAN_Discriminator()
    results = ConGAN_D.forward(img)
    # for i in results:
    #     print(i.shape)
    
    aug = ImgAugmentation()
    
    imgs = torch.randn(16, 3, 64, 64).to("cuda:0")
    print(f"imgs:{imgs.dtype} {imgs.shape} {imgs.device}")
    imgs_aug = aug.imgRealAugment(imgs)
    print(f"imgs:{imgs_aug.dtype} {imgs_aug.shape} {imgs_aug.device}")
    imgs_aug = aug.imgFakeAugment(imgs)
    print(f"imgs:{imgs_aug.dtype} {imgs_aug.shape} {imgs_aug.device}")
    
    