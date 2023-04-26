import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_shape = (1, 28, 28)

# 加载数据
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./My_Datasets/study', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)  # type: ignore

testset = torchvision.datasets.MNIST(root='./My_Datasets/study', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False) # type: ignore

# 定义编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        return x
    
encoder = Encoder().to(device)

# 定义解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        
    def forward(self, x):
        x = self.upsample1(torch.relu(self.conv1(x)))
        x = self.upsample2(torch.relu(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x
    
decoder = Decoder().to(device)

# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder(encoder, decoder).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
loss = nn.BCELoss()

# 训练自编码器
num_epochs = 50
for epoch in range(num_epochs):
    for data in trainloader:
        img, _ = data
        img = img.to(device)
        
        # 前向传播和反向传播
        output = autoencoder(img)
        l = loss(output, img)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, l.item()))
    
    