import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchinfo import summary
import matplotlib.pyplot as plt

from PIL import Image
import tqdm

#######################################################################################################
# CLASS: VAE
#######################################################################################################
class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 latent_dim: int,
                 input_size: int = 128,
                 hidden_dims: list = None,
                 **kwargs) -> None:
            
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
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
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        
        # Bulid Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)
        
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
    
    def encode(self, input: torch.Tensor) -> list[torch.Tensor]:
        result = self.Encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        result = self.Decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
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
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
        
    def sample(self, num_samples: int, current_device: str, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples
    
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

########################################################################################################
# CLASS: VAE TEST
########################################################################################################
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    vae_test = VAE(3, 128).cuda()
    
    optimizer = torch.optim.Adam(vae_test.parameters(), lr=1e-3, betas=[0.0, 0.9])
    
    for i in tqdm.trange(200):
        vae_test.train()
        
        train_loss = 0
        train_nsample = 0
        
        Y = vae_test(X)
        loss = vae_test.loss_function(*Y, **{'recons_weight': 0.5, 'kld_weight':0.5})
        loss['loss'].backward()
        
        optimizer.step()
        optimizer.zero_grad()
        Y = Y[0].detach().to('cpu').numpy()[0]
        Y = np.transpose(Y, (1, 2, 0))
        plt.imshow(Y)
        plt.savefig("test_vae_gen.png")
        plt.close()
        
# test()