import os
import torch
import torch.nn as nn
from torch.autograd import Variable

a = 1
b = [2, 3]

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

loss = nn.CrossEntropyLoss()
print(cross_entropy(y_hat, y), loss(y_hat, y))