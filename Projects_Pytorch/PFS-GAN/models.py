import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules as M
import torchvision.models as models

import numpy as np

channels = 3

class vgg(nn.Module):
    def __init__(self):
		super(vgg, self).__init__()
		v = models.vgg16(pretrained=True)
		self.m1 = v.features[:4]
		self.m2 = v.features[4:9]
		self.m3 = v.features[9:16]
		self.m4 = v.features[16:23]
    def forward(self, x):
		f0 = self.m1(x)
		f1 = self.m2(f0)
		f2 = self.m3(f1)
		f3 = self.m4(f2)
		return (f0, f1, f2, f3)


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

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

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

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
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x, feat=False):
		if feat:
			return self.model(x)
		else:
			return self.model(x) + self.bypass(x)

# # special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim):
		super(Generator, self).__init__()
		self.z_dim = z_dim

		self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
		nn.init.xavier_uniform(self.dense.weight.data, 1.)

		self.model0 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
		self.model1 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
		self.model2 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
		self.model3 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
		self.model4 = nn.BatchNorm2d(GEN_SIZE)
		self.model5 = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
		nn.init.xavier_uniform(self.model5.weight.data, 1.)
		self.model = nn.Sequential(self.model0,
									self.model1,
									self.model2,
									self.model3,
									self.model4,
									nn.ReLU(),
									self.model5,
									nn.Tanh(),
		)

    def forward(self, z):
		f1 = self.dense(z).view(-1, GEN_SIZE, 4, 4)
		out = self.model(f1)
		return out

class Discriminator_patch(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_patch, self).__init__()
        self.channels = channels
        self.inputBatch = nn.BatchNorm2d(channels)
        self.model0 = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.model1 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model4 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)

        self.conv1 = nn.Conv2d(DISC_SIZE, 1, 1, 1, 0)
        self.ReLU = nn.ReLU()
        self.avg = nn.AvgPool2d(4)

        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        
    def forward(self, x):       
        x = self.model0(x)
        x = self.model1(x)
        x = self.model2(x)

        x1 = self.conv1(x)
        x = self.model3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x2 = self.fc(x)
        return x1, x2

class Discriminator_UNIT(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_UNIT, self).__init__()
        self.channels = channels
        self.inputBatch = nn.BatchNorm2d(channels)
        self.model0 = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.model1 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)

        self.model = nn.Sequential(self.model0,
                                   self.model1,
                                   self.model2,
                                   self.model3,
                                   nn.ReLU(),
                                   nn.AvgPool2d(4),)

        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        #self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, DISC_SIZE)
        out = self.fc(f)
        return out

class Generator_UNIT(nn.Module):
    def __init__(self, z_dim):
        super(Generator_UNIT, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.front = nn.Sequential(
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2))
        self.back1 = nn.Sequential(
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
                nn.BatchNorm2d(GEN_SIZE),
                nn.ReLU(),
                nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1),
                nn.Tanh())
        self.back2 = nn.Sequential(
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
                ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
                nn.BatchNorm2d(GEN_SIZE),
                nn.ReLU(),
                nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1),
                nn.Tanh())

        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.back1[4].weight.data, 1.)
        nn.init.xavier_uniform(self.back2[4].weight.data, 1.)

    def forward(self, z, domain, feat=False, bp_single=True, t=False):
        f = self.front(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        if domain == 'S':
            return self.back1(f)
        elif domain == 'T':
            return self.back2(f)

class Encoder_UNIT(nn.Module):
    def __init__(self, channels=3):
        super(Encoder_UNIT, self).__init__()
        import torchvision.models as models

        self.frontA = models.vgg16(pretrained=True).features[:10]
        self.frontB = models.vgg16(pretrained=True).features[:10]
        self.back1 = models.vgg16(pretrained=True).features[10:]
        self.back2 = nn.Sequential(
                        nn.Linear(512*2*2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, DISC_SIZE),
                        nn.ReLU(),)

        self.mu = nn.Sequential(
                        nn.Conv2d(DISC_SIZE, DISC_SIZE, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(DISC_SIZE, DISC_SIZE, 3, 1, padding=1),)

        self.logvar = nn.Sequential(
                        nn.Conv2d(DISC_SIZE, DISC_SIZE, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(DISC_SIZE, DISC_SIZE, 3, 1, padding=1),)        
        
        nn.init.xavier_uniform(self.back2[0].weight.data, 1.)
        nn.init.xavier_uniform(self.back2[2].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[0].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[2].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[0].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[2].weight.data, 1.)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, domain):
        bsz = x.size(0)

        if domain == 'S':
            f = self.frontA(x)
        elif domain == 'T':
            f = self.frontB(x)

        f = self.back1(f).view(bsz, -1)
        f = self.back2(f).view(bsz, -1, 1, 1)
        mu, logvar = self.mu(f), self.logvar(f)
        z = self.reparameterize(mu, logvar)
        return mu.view(bsz, -1), logvar.view(bsz, -1), z.view(bsz, -1)


class Generator_cogan(nn.Module):
    def __init__(self, z_dim):
        super(Generator_cogan, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.front = nn.Sequential(
			ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
			ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2))
        self.back1 = nn.Sequential(
			ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
			ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
			nn.BatchNorm2d(GEN_SIZE),
			nn.ReLU(),
			nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1),
			nn.Tanh())
        self.back2 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1),
            nn.Tanh())

        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.back1[4].weight.data, 1.)
        nn.init.xavier_uniform(self.back2[4].weight.data, 1.)

    def forward(self, z, domain):
        f = self.front(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        if domain == 'S':
            return self.back1(f)
        elif domain == 'T':
            return self.back2(f)
            
class Discriminator_cogan(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_cogan, self).__init__()
        self.channels = channels
        self.front1 = nn.Sequential(
			FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
        	ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2))
        self.front2 = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2))

        self.back = nn.Sequential(
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(4))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)


    def forward(self, img, domain, feat=False):
        if domain == 'S':
            out = self.fc(self.back(self.front1(img)).view(-1, DISC_SIZE))
        elif domain == 'T':
            out = self.fc(self.back(self.front2(img)).view(-1, DISC_SIZE))
        return out

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
	self.channels = channels
	self.inputBatch = nn.BatchNorm2d(channels)
        self.model0 = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.model1 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)

        self.ReLU = nn.ReLU()
        self.avg = nn.AvgPool2d(4)

        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)

    def forward(self, x):
        f0 = self.model0(x)
        f1 = self.model1(f0)
        f2 = self.model2(f1)
        f3 = self.model3(f2)
        f3_r = self.ReLU(f3)
        f4 = self.avg(f3_r)
        f4_f = f4.view(-1, DISC_SIZE)
        out = self.fc(f4_f)
        return out


# class vgg(nn.Module):
#     def __init__(self):

# 	super(vgg, self).__init__()
# 	self.model = nn.Sequential(
# 		nn.Conv2d(3, 64, 3, 1, 1),
# 		nn.BatchNorm2d(64),
# 		nn.ReLU(inplace=True),
# 		nn.Conv2d(64, 64, 3, 1, 1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
# 		nn.MaxPool2d(2),
# 		nn.Conv2d(64, 128, 3, 1, 1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(128, 128, 3, 1, 1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
# 	)
#     def forward(self, x):

# 	return self.model(x)


class Relation(nn.Module):
    def __init__(self, channels=3, size_num=DISC_SIZE):
        super(Relation, self).__init__()

    def set(self, m):

        self.model_bn1_A = m.model_bn1[:10] 
        self.model_bn1_B = models.vgg16(pretrained=True).features[:10].cuda()
        for i in [0,2,5,7]:
            self.model_bn1_B[i].weight.data = m.model_bn1[i].weight.clone()
            self.model_bn1_B[i].bias.data = m.model_bn1[i].bias.clone()
        self.model_bn_share = m.model_bn1[10:] #.clone()
        self.model_bn2 = m.model_bn2
        self.mu = m.mu


    def forward(self, x1, x2):
        f1 = self.model_bn1_A(x1)
        f2 = self.model_bn1_B(x2)
#       print self.model_bn(x1).size()
        f1 = self.mu(self.model_bn2(self.model_bn_share(f1).view(x1.size(0), -1)).view(x1.size(0), -1, 1, 1))
        f2 = self.mu(self.model_bn2(self.model_bn_share(f2).view(x2.size(0), -1)).view(x2.size(0), -1, 1, 1))
        return ((f1-f2)**2).view(f1.size(0), -1) #torch.mean((f1-f2)**2, dim=1, keepdim=True)

class Encoder(nn.Module):
    def __init__(self, channels=3, size_num=DISC_SIZE):
        super(Encoder, self).__init__()
        self.model_bn1 =  models.vgg16(pretrained=True).features
        self.model_bn2 = nn.Sequential(
                nn.Linear(512*2*2, 1024),
                nn.ReLU(),
                nn.Linear(1024, size_num),
                nn.ReLU(),)

        self.mu = nn.Sequential(
                        nn.Conv2d(size_num, size_num, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(size_num, size_num, 3, 1, padding=1),)

        self.logvar = nn.Sequential(
                        nn.Conv2d(size_num, size_num, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(size_num, size_num, 3, 1, padding=1),)        
        
        nn.init.xavier_uniform(self.model_bn2[0].weight.data, 1.)
        nn.init.xavier_uniform(self.model_bn2[2].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[0].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[2].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[0].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[2].weight.data, 1.)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) #*0.1
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def forward(self, x):
        bsz = x.size(0)
        f = self.model_bn1(x).view(bsz, -1)
        f = self.model_bn2(f).view(bsz, -1, 1, 1)
        mu, logvar = self.mu(f), self.logvar(f)
        z = self.reparameterize(mu, logvar)
        return mu.view(bsz, -1), logvar.view(bsz, -1), z.view(bsz, -1)
