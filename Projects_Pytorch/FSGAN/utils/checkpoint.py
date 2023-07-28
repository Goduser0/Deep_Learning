from datetime import datetime
from matplotlib import lines
from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import os


def save_loss(checkpoint_path, epoch, n_epochs, batch, n_batchs, d_loss, g_loss):
    """write the D_loss and G_loss item to the loss_log.txt"""
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "loss_log.txt"), "a+") as log:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item = "%s Epoch%d/%d Batch%d/%d D_loss:%f G_loss:%f" % (time_str, epoch, n_epochs, batch, n_batchs, d_loss, g_loss)
        log.write(item + '\n')


def save_samples(sample_path, sample_interval, epoch, batch, n_batchs, fake_imgs_data):
    """save the samples during trainging to watch the process"""
    os.makedirs(sample_path, exist_ok=True)
    batches_done = epoch * n_batchs + batch
    if batches_done % sample_interval == 0:
        save_image(fake_imgs_data[:25],  sample_path + "/%d.png" % batches_done, nrow=5,normalize=True)


def plot_loss(log_path, mode="batch"):
    """plot D_loss and G_loss with epoch or batch `mode` according to the loss_log.txt"""
    plt.figure(figsize=(10, 5), dpi=200)
    batches = 0
    with open(log_path, 'r') as log:
        lines = log.readlines()
        n = len(lines)
        D_loss, G_loss = np.zeros(n), np.zeros(n)
        for i, line in tqdm(enumerate(lines)):
            item_list = line.split()
            D_loss[i] = float(item_list[-2].split(':')[-1])
            G_loss[i] = float(item_list[-1].split(':')[-1])
            # counter batch nums, equl to epoch1 times
            if item_list[2].split('/')[0][5:] == '1':
                batches += 1
    print(batches)
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
