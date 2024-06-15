import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as M
import torchvision
import torchvision.transforms as T
from torchinfo import summary

import warnings
warnings.filterwarnings("ignore")
#######################################################################################################
# CLASS: VariationalAutoEncoder
#######################################################################################################
class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 latent_dim: int,
                 input_size: int = 128,
                 hidden_dims: list = None,
                 **kwargs) -> None:
            
        super(VariationalAutoEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims
        
        # Bulid Encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
            
        self.Encoder = nn.Sequential(*encoder_layers)
        # FC
        self.last_layer_size = input_size // 2**(len(hidden_dims))
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.last_layer_size * self.last_layer_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.last_layer_size * self.last_layer_size, latent_dim)
        
        # Bulid Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.last_layer_size * self.last_layer_size)
        
        hidden_dims.reverse()
        decoder_layers = []
        
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.Decoder = nn.Sequential(*decoder_layers)
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),            
        )
    
    def encode(self, input: torch.Tensor):
        result = self.Encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.last_layer_size, self.last_layer_size)
        result = self.Decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs):
        """
        Returns:
            list[torch.Tensor]: [decode(z), input, mu, log_var, z]
            z是对mu, log_var重参化后的结构
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        recons_weight = kwargs['recons_weight']
        kld_weight = kwargs['kld_weight']
        
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = recons_weight * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD_loss': kld_loss.detach()}
        
    def sample(self, num_samples: int, current_device: str, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples
    
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]
    
    
#######################################################################################################
# CLASS: Encoder
#######################################################################################################
class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 latent_dim: int,
                 input_size: int = 128,
                 hidden_dims: list = None,
                 **kwargs) -> None:
        """
        return [mu, log_var, z]
        """
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims
        
        # Bulid Encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
            
        self.Encoder = nn.Sequential(*encoder_layers)
        # FC
        self.last_layer_size = input_size // 2**(len(hidden_dims))
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.last_layer_size * self.last_layer_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.last_layer_size * self.last_layer_size, latent_dim)
    
    def encode(self, input: torch.Tensor):
        result = self.Encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [mu, log_var, z]

#######################################################################################################
# CLASS: PFS_Encoder()
#######################################################################################################
class PFS_Encoder(nn.Module):
    def __init__(self, channels=3, latent_dim=128):
        super(PFS_Encoder, self).__init__()
        self.model_bn1 = torchvision.models.vgg16(pretrained=True).features
        self.model_bn2 = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.ReLU(),
        )
        
        self.mu = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, padding=1),
        )
        
        self.logvar = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, padding=1),
        )

        nn.init.xavier_uniform(self.model_bn2[0].weight.data, 1.)
        nn.init.xavier_uniform(self.model_bn2[2].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[0].weight.data, 1.)
        nn.init.xavier_uniform(self.mu[2].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[0].weight.data, 1.)
        nn.init.xavier_uniform(self.logvar[2].weight.data, 1.)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        bsz = x.size(0)
        f = self.model_bn1(x).view(bsz, -1)
        f = self.model_bn2(f).view(bsz, -1, 1, 1)
        mu, logvar = self.mu(f), self.logvar(f)
        z = self.reparameterize(mu, logvar)
        return mu.view(bsz, -1), logvar.view(bsz, -1), z.view(bsz, -1)
    
#######################################################################################################
# CLASS: UNIT_Encoder()
#######################################################################################################
    
class UNIT_Encoder(nn.Module):
    def __init__(self, channels=3):
        super(UNIT_Encoder, self).__init__()
        import torchvision.models as models

        self.frontA = models.vgg16(pretrained=True).features[:10]
        self.frontB = models.vgg16(pretrained=True).features[:10]
        self.back1 = models.vgg16(pretrained=True).features[10:]
        self.back2 = nn.Sequential(
                        nn.Linear(512*2*2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 128),
                        nn.ReLU(),)

        self.mu = nn.Sequential(
                        nn.Conv2d(128, 128, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, 1, padding=1),)

        self.logvar = nn.Sequential(
                        nn.Conv2d(128, 128, 3, 1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, 1, padding=1),)        
        
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

########################################################################################################
## FUNCTION: test()
########################################################################################################
def test_train_vae():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # dir = "My_Datasets/Classification/PCB-Crop/Mouse_bite/01_mouse_bite_06_1.jpg"
    dir = "My_Datasets/Classification/PCB-200/Open_circuit/000001_0_00_07022_09338.bmp"
    # dir = "My_Datasets/Classification/PCB-Crop/Spur/01_spur_01_0.jpg"
    img = Image.open(dir).convert("RGB")
    plt.imshow(img)
    plt.savefig("test_vae_raw.png")
    plt.close()
    trans = T.Compose([T.ToTensor(), T.Resize((128, 128))])
    X = trans(img).cuda()
    X = X.unsqueeze(0)
    print(X.shape)
    vae_test = VariationalAutoEncoder(3, 128).cuda()
    optimizer = torch.optim.Adam(vae_test.parameters(), lr=1e-3, betas=[0.0, 0.9])
    for i in tqdm.trange(200):
        vae_test.train()
        Y = vae_test(X)
        loss = vae_test.loss_function(*Y, **{'recons_weight': 0.5, 'kld_weight':0.5})
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        Y = Y[0].detach().to('cpu').numpy()[0]
        Y = np.transpose(Y, (1, 2, 0))
        plt.imshow(Y)
        plt.savefig("test_vae_gen.png")
        plt.close()

def test():
    X = torch.randn(8, 3, 128, 128)
    vae = Encoder(3, latent_dim=64)
    Y = vae(X)
    print(f"Input X:{X.shape}")
    print(f"Output Y:{Y[0].shape} {Y[1].shape} {Y[2].shape}")
    
if __name__ == "__main__":
    # test_train_vae()
    test()