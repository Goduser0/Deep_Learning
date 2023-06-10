# Compute MMD distance using pytorch
import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """ with multi RBF-bandwith Kernel to calculate
        Args:
            source (_type_): (sample_size_1, feature_size)
            target (_type_): (sample_size_2, feature_size)
            kernel_mul (float, optional): For calculating bandwith. Defaults to 2.0.
            kernel_num (int, optional): multi Kernel nums. Defaults to 5.
            fix_sigma (_type_, optional): whether to use fix sigma. Defaults to None.
        Returns:
            (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2) matrix.
                [   K_ss K_st
                    K_ts K_tt ]
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # concat
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # Gauss Kernel's |x - y|
        # calculate each kernel's bandwith
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        # exp(-|x-y|/bandwith), Formula for Gauss Kernel
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])  # K_ss
            YY = torch.mean(kernels[batch_size:, batch_size:])  # K_tt
            XY = torch.mean(kernels[:batch_size, batch_size:])  # K_st
            YX = torch.mean(kernels[batch_size:, :batch_size])  # K_ts
            loss = torch.mean(XX + YY - XY - YX)
            return loss


if __name__ == '__main__':
    from swd import read_images

    MMDLoss = MMD_loss('rbf')
    real_path = "../dataset/NEU/NEU-50-r64/train/Cr"
    fake_path = "../work_dir/generator/wgan-gp/Cr/epoch10000"
    real_images = read_images(real_path)
    fake_images = read_images(fake_path)

    a = real_images.reshape(real_images.shape[0], -1)
    b = fake_images.reshape(fake_images.shape[0], -1)
    print(MMDLoss(a, b))