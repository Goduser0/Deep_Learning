import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 加载数据集
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_dataset = datasets.CIFAR10(root='./My_Datasets/study', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./My_Datasets/study', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4) # type: ignore
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4) # type: ignore

# 定义模型，加载预训练参数
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
def train(model, train_loader, criterion, optimizer, device=torch.device('cuda:0')):
    model.to(device)
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
def test(model, test_loader, criterion, device=torch.device('cuda:0')):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_corrects = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds == labels.data)
        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = float(total_corrects) / len(test_loader.dataset)
    return avg_loss, accuracy

# 训练和测试模型
num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test(model, test_loader, criterion)
    print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_acc))

# 保存模型
torch.save(model.state_dict(), 'cifar10_resnet18.pth')