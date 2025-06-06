import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# 定义判别器    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# 超参数定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-5
batch_size = 128
hidden_size = 256
input_size = 100
num_epochs = 100

# 定义数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
    
train_dataset = MNIST(root='./My_Datasets/study', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 初始化生成器和判别器
generator = Generator(input_size, hidden_size, 784).to(device)
discriminator = Discriminator(784, hidden_size).to(device)

# 定义损失函数与优化器
loss = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

G_loss_list_list = []
D_loss_list_list = []
# train
for epoch in range(num_epochs):
    G_loss_list = []
    D_loss_list = []
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(-1, 784).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 训练判别器
        d_optimizer.zero_grad()
        
        # 训练判别器使用真实数据
        real_outputs = discriminator(real_images)
        d_loss_real = loss(real_outputs, real_labels)
        d_loss_real.backward()
        
        # 训练判别器使用生成数据
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        d_loss_fake = loss(fake_outputs, fake_labels)
        d_loss_fake.backward()
        
        d_loss = d_loss_real + d_loss_fake
        D_loss_list.append(d_loss.item())
        d_optimizer.step()
        
        # 训练生成器
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        g_loss = loss(fake_outputs, real_labels)
        G_loss_list.append(g_loss.item())
        g_loss.backward()
        g_optimizer.step()
        
        # if (i+1) % 200 == 0:
        #     print("Epoch [{}/{}], Step[{}/{}], d_loss:{:.4f}, g_loss:{:.4f}".format(epoch+1, num_epochs, i+1, len(train_loader), d_loss.item(), g_loss.item()) )
    
    G_loss_list_list.append(np.mean(G_loss_list))
    D_loss_list_list.append(np.mean(D_loss_list))
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
    ax1.plot([y for y in G_loss_list_list], label="g_loss")
    ax1.legend()
    fig.tight_layout()
    plt.savefig('g_loss.jpg')
    plt.close()     
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
    ax1.plot([y for y in D_loss_list_list], label="d_loss")
    ax1.legend()
    fig.tight_layout()
    plt.savefig('d_loss.jpg')
    plt.close() 