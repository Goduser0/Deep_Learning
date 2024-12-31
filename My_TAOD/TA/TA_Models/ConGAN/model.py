import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm

########################################################################################################
# CLASS: ConGAN_Generator()
########################################################################################################
class ConGAN_Generator(nn.Module):
    """
    forward:
        input: [batch_size, 100, 1, 1]
        output:[batch_size, 3, 32, 32]
    """
    def __init__(self, output_channels=3):
        super().__init__()
        
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=output_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.output = nn.Tanh()
        
    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x

########################################################################################################
# CLASS: ConGAN_Discriminator()
########################################################################################################
class ConGAN_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, 32, 32] 
        outpue:[batch_size, 1, 1, 1]
    feature_extraction:
        input: [batch_size, 3, 32, 32] 
        outpue:[batch_size, 1024*4*4]
    """
    
    def __init__(self, input_channels=3):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x
    
    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


if __name__ == "__main__":
    z = torch.randn(16, 100, 1, 1)
    DCGAN_G = DCGAN_Generator(output_channels=3)
    print(DCGAN_G.forward(z).shape)
    
    img = torch.randn(16, 3, 32, 32)
    DCGAN_D = DCGAN_Discriminator(input_channels=3)
    print(DCGAN_D.forward(img).shape)
    print(DCGAN_D.feature_extraction(img).shape)