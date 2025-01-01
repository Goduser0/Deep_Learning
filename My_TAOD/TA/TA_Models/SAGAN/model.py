import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

########################################################################################################
# CLASS: SpectralNorm()
########################################################################################################
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    
    def l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)
    
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
########################################################################################################
# CLASS: Self_Attn()
########################################################################################################
class Self_Atten(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Atten, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        
        output = torch.bmm(proj_value, attention.permute(0, 2, 1))
        output = output.view(m_batchsize, C, width, height)
        output = self.gamma*output + x
        return output, attention

########################################################################################################
# CLASS: SAGAN_Generator()
########################################################################################################
class SAGAN_Generator(nn.Module):
    """
    forward:
        input: [batch_size, 100]
        output:
            [batch_size, 3, 64, 64]
            [batch_size, 256, 256]
            [batch_size, 1024, 1024]
    """
    def __init__(self, img_size=64, z_dim=100, conv_dim=64):
        super(SAGAN_Generator, self).__init__()
        self.imgsize = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        
        repeat_num = int(np.log2(self.imgsize)) - 3
        mult = 2 ** repeat_num
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim*mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim*mult))
        layer1.append(nn.ReLU())
        
        curr_dim = conv_dim*mult
        
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())
        
        curr_dim = int(curr_dim / 2)
        
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))        
        layer3.append(nn.ReLU())
        
        if self.imgsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)
        
        self.attn1 = Self_Atten(128, 'relu')
        self.attn2 = Self_Atten(64, 'relu')
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        
        return out, p1, p2

########################################################################################################
# CLASS: SAGAN_Discriminator()
########################################################################################################
class SAGAN_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, 64, 64] 
        output:
            [batch_size]
            [batch_size, 64, 64]
            [batch_size, 16, 16]
    """
    def __init__(self, img_size=64, conv_dim=64):
        super(SAGAN_Discriminator, self).__init__()
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        
        curr_dim = conv_dim
        
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim*2
        
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim*2
        
        if self.img_size == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        
        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)
        
        self.attn1 = Self_Atten(256, 'relu')
        self.attn2 = Self_Atten(512, 'relu')
        
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        return out.squeeze(), p1, p2


if __name__ == "__main__":
    z = torch.randn(16, 100)
    DCGAN_G = SAGAN_Generator(16)
    print(DCGAN_G.forward(z)[0].shape)
    print(DCGAN_G.forward(z)[1].shape)
    print(DCGAN_G.forward(z)[2].shape)
    
    img = torch.randn(16, 3, 64, 64)
    DCGAN_D = SAGAN_Discriminator()
    print(DCGAN_D.forward(img)[0].shape)
    print(DCGAN_D.forward(img)[1].shape)
    print(DCGAN_D.forward(img)[2].shape)