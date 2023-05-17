import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np


dataset = pd.read_table('./1.txt', sep = '\t', names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])
dataset.to_csv("dataset.csv")

X = torch.tensor(dataset.iloc[:,:4].values)
y = torch.tensor(dataset.iloc[:,-1].values.reshape(-1, 1))
X = X.to(torch.float32)
y = y.to(torch.float32)
data_arrays = (X, y)

dataset = data.TensorDataset(*data_arrays)
train_ratio = 0.5
val_ratio = 1 - train_ratio
train_data, test_data = data.random_split(dataset, [int(len(dataset) * train_ratio), int(len(dataset) * val_ratio)])

train_iter = data.DataLoader(train_data, batch_size=1, shuffle=True)
test_iter = data.DataLoader(test_data, batch_size=1, shuffle=False)

class Model(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        y_pred = self.net(x)
        return y_pred

net = Model(4, 16, 1)
loss = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 10

# train
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)
        optim.zero_grad()
        l.backward()
        optim.step()
    
# vaild
for batch, (X, y) in enumerate(test_iter):
    for (XX, yy) in zip(X, y):
        y_pred = net(XX)
        y_pred = y_pred.data.item()
        
        if y_pred >= 0.5:
            yy_pred = 1
        else:
            yy_pred = 0
        
        right=0.0
        false=0.0
        if yy_pred == y:
            right+=1
        else:
            false+=1
print(f"acc={right/(right+false)}")