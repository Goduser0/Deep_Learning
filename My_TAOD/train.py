import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class Accumulator:
    """累加器"""
    def __init__(self, n):
        # 创建len=n的list
        self.data = [0.0]*n
        
    def add(self, *args):
        # data累加args
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 重置累加器
        self.data = [0.0]*len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class Timer():
    """计时器"""
    def __init__(self):
        self.times=[]
        self.start()
        
    def start(self):
        # 记录开始时间戳
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()    

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def cal_correct(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)

    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter, device=None):
    """计算模型在数据集上的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(cal_correct(net(X), y), size(y))
    return metric[0] / metric[1]
    
def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    print('training on', device)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fuction = torch.nn.CrossEntropyLoss()
    
    timer = Timer()
    for epoch in tqdm(range(num_epochs)):
        metric = Accumulator(3)
        net.train()
        
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = loss_fuction(y_hat, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                metric.add(loss*X.shape[0], cal_correct(y_hat, y),X.shape[0])
            timer.stop()
            
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)
    print(f'loss{train_loss:.3f}, train acc{train_acc:.3f}, '
        f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec'
        f'on {str(device)}')