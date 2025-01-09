import torch
import torch.nn as nn
import torch.nn.functional as F

import random

########################################################################################################
# CLASS: ResidualBlock()
########################################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

########################################################################################################
# CLASS: CycleGAN_Generator()
########################################################################################################
class CycleGAN_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(CycleGAN_Generator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7, padding=0),
            nn.InstanceNorm2d(64), 
            nn.ReLU(inplace=True),
        ]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2 
        
        for _ in range(n_residual_blocks):
            model += [
                ResidualBlock(in_features),
            ]
        
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        
        model += [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(64, output_nc, 7), 
            nn.Tanh(), 
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        input: [B, C, img_size, img_size]
        output:[B, C, img_size, img_size]
        """
        return self.model(x)

########################################################################################################
# CLASS: CycleGAN_Discriminator()
########################################################################################################
class CycleGAN_Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(CycleGAN_Discriminator, self).__init__()

        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        input: [B, 3, img_size, img_size]
        output:[B, 1]
        """
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])

#######################################################################################################
# FUNCTION: Weights_Init_Normal()
#######################################################################################################
def Weights_Init_Normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

#######################################################################################################
# class: Lambda_Learing_Rate()
#######################################################################################################
class Lambda_Learing_Rate():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
#######################################################################################################
# CLASS: Replay_Buffer()
#######################################################################################################
class Replay_Buffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))

if __name__ == "__main__":
    z = torch.randn(16, 3, 64, 64)
    G = CycleGAN_Generator(3, 3)
    img = G.forward(z)
    print(img.shape)
    D = CycleGAN_Discriminator(3)
    label = D.forward(img)
    print(label.shape)