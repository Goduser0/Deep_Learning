import torch
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from torch import autograd
from torch import tensor


def img_uint2tensor(img_uint):
    """ trans img_unit like [N, H, W, C], 0-255 to tensor
        like [N, C, H, W], -1-1.
        Keep the device.
    """
    return img_uint.permute(0, 3, 1, 2) / 127.5 - 1.


def img_tensor2uint(img_tensor):
    """ trans img_tensor like [N, C, H, W], -1-1 to unit
        like [N, H, W, C], 0-255.
        Keep the device.
    """
    return ((img_tensor + 1.) * 127.5).permute(0, 2, 3, 1).to(torch.uint8)


def save_loss(checkpoint_path, epoch, n_epochs, batch, n_batchs, GAN_G_loss, GAN_Feature_Match_loss, MMD_Thriplet_loss, g_loss, d_loss):
    """write the D_loss and G_loss item to the loss_log.txt"""
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "loss_log.txt"), "a+") as log:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item = "%s Epoch%d/%d Batch%d/%d GAN_G_loss:%f GAN_Feature_Match_loss:%f MMD_Thriplet_loss:%f G_loss:%f D_loss:%f" % (time_str, epoch, n_epochs, batch, n_batchs, GAN_G_loss, GAN_Feature_Match_loss, MMD_Thriplet_loss, g_loss, d_loss)
        log.write(item + '\n')


def plot_loss(log_path, mode="batch"):
    """plot D_loss and G_loss with epoch or batch `mode` according to the loss_log.txt"""
    plt.figure(figsize=(10, 5), dpi=200)
    batches = 0
    with open(log_path, 'r') as log:
        lines = log.readlines()
        n = len(lines)
        D_loss, G_loss = np.zeros(n), np.zeros(n)
        for i, line in enumerate(lines):
            item_list = line.split()
            D_loss[i] = float(item_list[-1].split(':')[-1])
            G_loss[i] = float(item_list[-2].split(':')[-1])
            # counter batch nums, equl to epoch1 times
            if item_list[2].split('/')[0][5:] == '1':
                batches += 1
    # print(batches)
    if mode == "batch":
        # Each Batch plot a loss
        steps = list(range(n))
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.plot(steps, D_loss, label='discriminator_loss')
        plt.plot(steps, G_loss, 'r', label='generator_loss')
        plt.legend(loc=1)
        plt.show()
        fig_path = os.path.dirname(os.path.abspath(log_path)) +  "/loss_batch.png"
        plt.savefig(fig_path, dpi=200)
    if mode == "epoch":
        # Each Epoch plot a loss
        epochs = list(range(n // batches))
        D_loss = D_loss.reshape(-1, batches).mean(axis=-1)  # mean loss for one epoch
        G_loss = G_loss.reshape(-1, batches).mean(axis=-1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, D_loss, label='discriminator_loss')
        plt.plot(epochs, G_loss, 'r', label='generator_loss')
        plt.legend(loc=1)
        plt.show()
        fig_path = os.path.dirname(os.path.abspath(log_path)) +  "/loss_epoch.png"
        plt.savefig(fig_path, dpi=200)


def gradient_penalty(loss_func, discriminator, real_input, fake_input):
    """Calcaulate the gradient penalty loss for wgan-gp / wgan-lp / dragan"""

    if loss_func == "wgan-gp" or loss_func == "wgan-lp":
        # random weight for interpolation between real and fake images `input`
        # x_inter = alpha * x_real + (1-alpha) * x_fake, alpha belongs [0, 1] as uniform
        alpha = torch.rand(real_input.shape[0], 1, 1, 1).cuda().expand_as(real_input)  # [b, 1, 1, 1] => [b, h, w, c]
        interpolated = (alpha * real_input + (1 - alpha) * fake_input).requires_grad_(True)
        # calculate D's gradient to interpolated
        out = discriminator(interpolated)[-2]
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

        out = discriminator(interpolated)[-2]
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