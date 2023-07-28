import torch
import torch.nn as nn
import numpy as np
from torch import autograd
from torch import tensor

# TODO 生成一个预训练的生成器网络和判别器网络，用全部的图片，作为半监督

def discriminator_loss(loss_func, real_output, fake_output):
    real_loss = 0
    fake_loss = 0

    if loss_func == "gan" or loss_func == "dragan":
        # BCEWithLogitsLoss() have `Sigmoid`, ones/zeros_like keep data type and device
        # reference : `Sphere GAN` code
        real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
        fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))

    if loss_func == "wgan-gp" or loss_func == "wgan-lp":
        real_loss = -torch.mean(real_output)
        fake_loss = torch.mean(fake_output)

    if loss_func == "lsgan":
        real_loss = torch.mean(torch.square(real_output - torch.ones_like(real_output)))
        fake_loss = torch.mean(torch.square(fake_output))

    if loss_func == "hinge":
        real_loss = torch.mean(nn.ReLU()(1.0 - real_output))
        fake_loss = torch.mean(nn.ReLU()(1.0 + fake_output))

    if loss_func == "rgan" or loss_func == "ragan":
        # ragan use Relativistic Discriminator's mean nother random samples
        real_ = torch.ones_like(real_output)
        fake_ = torch.zeros_like(fake_output)
        if loss_func == "ragan":
            real_loss = nn.BCEWithLogitsLoss()(real_output - fake_output.mean(0, keepdim=True), real_)
            fake_loss = nn.BCEWithLogitsLoss()(fake_output - real_output.mean(0, keepdim=True), fake_)
        elif loss_func == "rgan":
            real_loss = nn.BCEWithLogitsLoss()(real_output - fake_output, real_)
            fake_loss = nn.BCEWithLogitsLoss()(fake_output - real_output, fake_)

    if loss_func == "sphere":
        sphere_loss = HyperSphereLoss()
        real_loss = sphere_loss(real_output)
        fake_loss = -sphere_loss(fake_output)

    d_loss = real_loss + fake_loss  # not divide 2
    return d_loss


def generator_loss(loss_func, fake_output, real_output=None):
    """only ragn/rsgan need real_output = real_images_out"""
    fake_loss = 0

    if loss_func == "gan" or loss_func == "dragan":
        fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))

    if loss_func == "wgan-gp" or loss_func == "wgan-lp":
        fake_loss = -torch.mean(fake_output)

    if loss_func == "lsgan":
        fake_loss = torch.mean(torch.square(fake_output - torch.ones_like(fake_output)))

    if loss_func == "hinge":
        fake_loss = -torch.mean(fake_output)

    if loss_func == "rgan" or loss_func == "ragan":
        real_ = torch.ones_like(fake_output)
        if loss_func == "ragan":
            fake_loss = nn.BCEWithLogitsLoss()(fake_output - real_output.mean(0, keepdim=True), real_)
        elif loss_func == "rgan":
            fake_loss = nn.BCEWithLogitsLoss()(fake_output - real_output, real_)

    if loss_func == "sphere":
        sphere_loss = HyperSphereLoss()
        fake_loss = sphere_loss(fake_output)

    g_loss = fake_loss
    return g_loss


def gradient_penalty(loss_func, discriminator, real_input, fake_input):
    """Calcaulate the gradient penalty loss for wgan-gp / wgan-lp / dragan"""

    if loss_func == "wgan-gp" or loss_func == "wgan-lp":
        # random weight for interpolation between real and fake images `input`
        # x_inter = alpha * x_real + (1-alpha) * x_fake, alpha belongs [0, 1] as uniform
        alpha = torch.rand(real_input.shape[0], 1, 1, 1).cuda().expand_as(real_input)  # [b, 1, 1, 1] => [b, h, w, c]
        interpolated = (alpha * real_input + (1 - alpha) * fake_input).requires_grad_(True)
        # calculate D's gradient to interpolated
        out = discriminator(interpolated)
        grad = autograd.grad(
            outputs=out,
            inputs=interpolated,
            grad_outputs=torch.ones(out.shape).cuda(),  # requires_grad = False
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        grad = grad.reshape(grad.shape[0], -1)  # gradient of D(interpolated) and flatten
        grad_norm = grad.norm(2, dim=1)

        GP = 0
        if loss_func == "wgan-gp":
            GP = torch.mean(torch.square((grad_norm - 1.)))
        if loss_func == "wgan-lp":
            GP = torch.mean(torch.square(torch.maximum(tensor(0.0).cuda(), grad_norm - 1.)))
        return GP
    
    elif loss_func == "dragan":
        # random wright only based on real_input
        alpha = torch.rand(real_input.shape[0], 1, 1, 1).cuda().expand_as(real_input)  # [b, h, w, c]
        eps = torch.rand(real_input.size())
        std = real_input.std()
        noise = 0.5 * std * eps  # delta in paper
        interpolated = alpha * real_input + (1 - alpha) * (real_input + noise)

        out = discriminator(interpolated)
        grad = autograd.grad(
            outputs=out,
            inputs=interpolated,
            grad_outputs=torch.ones(out.shape).cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        grad = grad.reshape(grad.shape[0], -1)
        grad_norm = grad.norm(2, dim=1)

        GP = torch.mean(torch.square(grad_norm - 1.))
        return GP


class HyperSphereLoss(nn.Module):
    """SphereGAN Loss"""
    def project_to_hypersphere(self, p):
        """ Project `p` to the sphere points `x & y`.
            Following formula: (p, 0) -> x=2p/(p^2+1), y=(p^2-1)/(p^2+1)
            real/fake output from (B, c) ro (B, c+1).
        """
        p_norm = torch.norm(p, dim=1, keepdim=True) ** 2
        x = 2 * p / (p_norm + 1)
        y = (p_norm - 1) / (p_norm + 1)
        return torch.cat([x, y], dim=1)

    def forward(self, input, moment=3):
        '''
        Calcuate distance between input and N(North Pole, represents real label) using hypersphere metrics. Matching monents from 1-3.
        ds(p, q) = acos((p2p2 -p2 -q2 +4pq + 1) / (p2+1)(q2+1)).
        when q is N(north pole), simplify as:
        ds(p, N) = acos(4pq / 2(p2+1)) = acos(2p / p2+1)
        '''
        p = self.project_to_hypersphere(input)
        p_norm = torch.norm(p, dim=1) ** 2
        sphere_d = torch.acos((2 * p[:, -1]) / (1 + p_norm))  # calcuate sphere_d between p and real
        loss = 0
        for i in range(1, moment + 1):
            loss += torch.mean(torch.pow(sphere_d, i))
        return loss