import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

########################################################################################################
# CLASS: WGAN_GP_Generator()
########################################################################################################
class WGAN_GP_Generator(nn.Module):
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
# CLASS: WGAN_GP_Discriminator()
########################################################################################################
class WGAN_GP_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, 32, 32] 
        output:[batch_size, 1, 1, 1]
    feature_extraction:
        input: [batch_size, 3, 32, 32] 
        output:[batch_size, 1024*4*4]
    """
    
    def __init__(self, input_channels=3):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
        )
        
    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x
    
    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


########################################################################################################
# FUNCTION: Cal_Gradient_Penalty()
########################################################################################################
def Cal_Gradient_Penalty(discriminator, real_imgs, fake_imgs, device):
    batch_size = real_imgs.shape[0]
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).to(device).expand_as(real_imgs)
    
    interpolated = eta*real_imgs + (1-eta)*fake_imgs.requires_grad_(True)

    prob_interpolated = discriminator(interpolated)
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

if __name__ == "__main__":
    # G
    z = torch.randn(16, 100, 1, 1)
    WGAN_GP_G = WGAN_GP_Generator(output_channels=3)
    print(WGAN_GP_G.forward(z).shape)
    # D
    img = torch.randn(16, 3, 32, 32)
    WGAN_GP_D = WGAN_GP_Discriminator(input_channels=3)
    print(WGAN_GP_D.forward(img).shape)
    print(WGAN_GP_D.feature_extraction(img).shape)
    # GP
    
    WGAN_GP_D.to("cuda:0")
    real_imgs = torch.FloatTensor(16, 3, 32, 32).uniform_(0, 1).to("cuda:0")
    fake_imgs = torch.FloatTensor(16, 3, 32, 32).uniform_(0, 1).to("cuda:0")
    GP = Cal_Gradient_Penalty(WGAN_GP_D, real_imgs, fake_imgs, "cuda:0")
    print(GP)