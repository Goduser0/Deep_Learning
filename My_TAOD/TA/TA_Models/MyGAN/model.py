import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torch import from_numpy
from torch import autograd

########################################################################################################
# CLASS: SelfAttention()
########################################################################################################
class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation="relu", k=8):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W*H)
        qk = torch.bmm(proj_query, proj_key)
        attention = self.softmax(qk)
        proj_value = self.value(x).view(B, -1, W*H)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out, attention

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
# CLASS: MyGAN_Encoder()
########################################################################################################
class MyGAN_Encoder(nn.Module):
    def __init__(self, input_nc=3, n_residual_blocks=9):
        super(MyGAN_Encoder, self).__init__()
        
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, 8, 7, padding=0),
            nn.InstanceNorm2d(8), 
            nn.ReLU(inplace=True),
        ]
        
        in_features = 8
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, 7, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2 
        
        for _ in range(n_residual_blocks):
            model += [
                ResidualBlock(in_features),
            ]
        
        model += [
            nn.AvgPool2d(4),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        input: [B, C, 64, 64]
        output:[B, 64]
        """
        return self.model(x)


########################################################################################################
# CLASS: MyGAN_Generator()
########################################################################################################
class MyGAN_Generator(nn.Module):
    """
    forward:
        input: [batch_size, 128]
        output:[batch_size, 3, 64, 64]
    """
    def __init__(self, img_size=64, z_dim=128, conv_dim=64):
        super(MyGAN_Generator, self).__init__()
        
        self.img_size = img_size
        layer0 = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        repeat_num = int(np.log2(self.img_size)) - 3
        multi = 2 ** repeat_num
        
        layer0.append(nn.Linear(128, 128))
        layer0.append(nn.ReLU())
        
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim*multi, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim*multi))
        layer1.append(nn.ReLU())
        
        curr_dim = conv_dim * multi
        
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim // 2))
        layer2.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(curr_dim // 2))
        layer3.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(curr_dim // 2))
        layer4.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn2_dim = curr_dim
        
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        
        self.l0 = nn.Sequential(*layer0)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
        
    def forward(self, z):
        z = self.l0(z)
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        return out

########################################################################################################
# CLASS: MyGAN_Discriminator()
########################################################################################################
class MyGAN_Discriminator(nn.Module):
    """
    forward:
        input: [batch_size, 3, 64, 64] 
        output:
        [
            [batch_size, 64, 32, 32],
            [batch_size, 128, 16, 16],
            [batch_size, 256, 8, 8],
            [batch_size, 512, 4, 4],
            [batch_size, 1],
            [batch_size, 64],
        ]
    """
    def __init__(self, img_size=64, conv_dim=64):
        super(MyGAN_Discriminator, self).__init__()
        self.img_size = img_size
        layer1 = [] 
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attn2_dim = curr_dim
        
        last.append(nn.Conv2d(curr_dim, 1, 4))
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
        
        self.MLP = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        
        self.u1 = nn.ConvTranspose2d(512, 512 // 2, 4, 2, 1)
        self.u2 = nn.ConvTranspose2d(256, 256 // 2, 4, 2, 1)
        self.u3 = nn.ConvTranspose2d(128, 128 // 2, 4, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.w4 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
                
    def forward(self, x):
        results = []
        out = self.l1(x)
        feature1 = out
        results.append(out)

        out = self.l2(out)
        feature2 = out
        results.append(out)
        
        out = self.l3(out)
        out, p1 = self.attn1(out)
        feature3 = out
        results.append(out)
        
        out = self.l4(out)
        out, p2 = self.attn2(out)
        feature4 = out
        results.append(out)
        
        out = self.last(out)
        out = out.reshape(out.shape[0], -1)
        results.append(out)
        
        out = self.MLP(out)
        results.append(out)
        
        feature4_up = self.u1(feature4)
        feature3 = feature3 + feature4_up
        feature3_up = self.u2(feature3)
        feature2 = feature2 + feature3_up
        feature2_up = self.u3(feature2)
        feature1 = feature1 + feature2_up
        feat_avg = self.avg(feature1).reshape(feature1.shape[0], -1)
        feat_attn = self.w4(feat_avg)
        feat_attn = self.softmax(feat_attn)
        
        for i in range(feat_attn.shape[1]):
            attn = feat_attn[:, i].reshape(feat_attn.shape[0], 1, 1, 1)
            results[i] = attn * results[i]

        return results


########################################################################################################
# FUNCTION: Cal_Gradient_Penalty()
########################################################################################################
def Cal_Gradient_Penalty(discriminator, real_imgs, fake_imgs, device):
    batch_size = real_imgs.shape[0]
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).to(device).expand_as(real_imgs)
    
    interpolated = eta*real_imgs + (1-eta)*fake_imgs.requires_grad_(True)
    prob_interpolated = discriminator(interpolated)[-2]
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
    MyGAN_G = MyGAN_Encoder()
    z = torch.randn(16, 3, 64, 64)
    print(MyGAN_G.forward(z).shape)
    
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
# FUNCTION: Weights_Init_Normal()
#######################################################################################################
def Weights_Init_Normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

########################################################################################################
# FUNCTION: Cal_Gradient_Penalty()
########################################################################################################
def Cal_Gradient_Penalty(discriminator, real_imgs, fake_imgs, device):
    batch_size = real_imgs.shape[0]
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).to(device).expand_as(real_imgs)
    
    interpolated = eta*real_imgs + (1-eta)*fake_imgs.requires_grad_(True)
    prob_interpolated = discriminator(interpolated)[-2]
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
