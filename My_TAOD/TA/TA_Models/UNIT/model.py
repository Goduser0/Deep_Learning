import numpy as np

import torch
import torch.nn as nn
import torch.nn.modules as M
import torchvision

########################################################################################################
# CLASS: ResBlockGenerator()
########################################################################################################
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            M.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            M.InstanceNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)
    

########################################################################################################
# CLASS: UNIT_Generator()
########################################################################################################
class UNIT_Generator(nn.Module):
    """
    forward:
        input: [batch_size, z_dim]
        output:[batch_size, 3, img_size, img_size]
    """
    def __init__(self, z_dim, img_size):
        super(UNIT_Generator, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.dense = nn.Linear(self.z_dim, 4 * 4 * img_size)
        self.front = nn.Sequential(
                ResBlockGenerator(img_size, img_size, stride=2),
                ResBlockGenerator(img_size, img_size, stride=2))
        self.back1 = nn.Sequential(
                ResBlockGenerator(img_size, img_size, stride=2),
                ResBlockGenerator(img_size, img_size, stride=2),
                nn.BatchNorm2d(img_size),
                nn.ReLU(),
                nn.Conv2d(img_size, 3, 3, stride=1, padding=1),
                nn.Tanh())
        self.back2 = nn.Sequential(
                ResBlockGenerator(img_size, img_size, stride=2),
                ResBlockGenerator(img_size, img_size, stride=2),
                nn.BatchNorm2d(img_size),
                nn.ReLU(),
                nn.Conv2d(img_size, 3, 3, stride=1, padding=1),
                nn.Tanh())

        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.back1[4].weight.data, 1.)
        nn.init.xavier_uniform_(self.back2[4].weight.data, 1.)

    def forward(self, z, domain, feat=False, bp_single=True, t=False):
        f = self.front(self.dense(z).view(-1, self.img_size, 4, 4))
        if domain.lower() == 'src':
            return self.back1(f)
        elif domain.lower() == 'tar':
            return self.back2(f)

########################################################################################################
# CLASS: FirstResBlockDiscriminator()
########################################################################################################
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2),
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )
        
    def forward(self, x):
        return self.model(x) + self.bypass(x)
    

########################################################################################################
# CLASS: ResBlockDiscriminator()
########################################################################################################
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x, feat=False):
        if feat:
            return self.model(x)
        else:
            return self.model(x) + self.bypass(x)

########################################################################################################
# CLASS: UNIT_Discriminator()
########################################################################################################
class UNIT_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, img_size, img_size]
        output:[batch_size, 1]
    """
    def __init__(self, channels=3, img_size=64):
        super(UNIT_Discriminator, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.inputBatch = nn.BatchNorm2d(channels)
        self.layer0 = FirstResBlockDiscriminator(channels, img_size, stride=2)
        self.layer1 = ResBlockDiscriminator(img_size, img_size, stride=2)
        self.layer2 = ResBlockDiscriminator(img_size, img_size, stride=2)
        self.layer3 = ResBlockDiscriminator(img_size, img_size, stride=2)
        
        self.model = nn.Sequential(
            self.layer0,
            self.layer1, 
            self.layer2,
            self.layer3, 
            nn.ReLU(), 
            nn.AvgPool2d(4),
        )
        self.fc = nn.Linear(img_size, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
    
    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.img_size)
        out = self.fc(f)
        return out

########################################################################################################
# CLASS: UNIT_Encoder()
########################################################################################################
class UNIT_Encoder(nn.Module):
    """
    forward:
        input: [batch_size, 3, img_size, img_size]
        output:
        [
            [batch_size, img_size],
            [batch_size, img_size],
            [batch_size, img_size],
        ]
    """
    def __init__(self, channels=3, img_size=64):
        super(UNIT_Encoder, self).__init__()
        self.frontA = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:10]
        self.frontB = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:10]
        self.back1 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[10:]
        self.back2 = nn.Sequential(
                        nn.Linear(512*2*2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, img_size),
                        nn.ReLU(),)

        self.mu = nn.Sequential(
                        nn.Conv2d(img_size, img_size, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(img_size, img_size, 3, 1, padding=1),)

        self.logvar = nn.Sequential(
                        nn.Conv2d(img_size, img_size, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(img_size, img_size, 3, 1, padding=1),)        
        
        nn.init.xavier_uniform_(self.back2[0].weight.data, 1.)
        nn.init.xavier_uniform_(self.back2[2].weight.data, 1.)
        nn.init.xavier_uniform_(self.mu[0].weight.data, 1.)
        nn.init.xavier_uniform_(self.mu[2].weight.data, 1.)
        nn.init.xavier_uniform_(self.logvar[0].weight.data, 1.)
        nn.init.xavier_uniform_(self.logvar[2].weight.data, 1.)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, domain):
        bsz = x.size(0)

        if domain.lower() == 'src':
            f = self.frontA(x)
        elif domain.lower() == 'tar':
            f = self.frontB(x)

        f = self.back1(f).view(bsz, -1)
        f = self.back2(f).view(bsz, -1, 1, 1)
        mu, logvar = self.mu(f), self.logvar(f)
        z = self.reparameterize(mu, logvar)
        return mu.view(bsz, -1), logvar.view(bsz, -1), z.view(bsz, -1)
