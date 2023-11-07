import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from d2l import torch as d2l
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torch.backends import cudnn


###########################################################################################################
# 通用函数
###########################################################################################################
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

class Timer(object):
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


def cal_correct(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)

    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的准确率"""
    if isinstance(net, nn.Module):
        net.eval()
    
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.cuda() for x in X]
            else:
                X = X.cuda()
            y = y.cuda()
            metric.add(cal_correct(net(X), y), size(y))
    return metric[0] / metric[1]

def save_results(config, content, plot=False):
    """将实验数据存为csv文件"""
    assert os.path.exists(config.result_dir), f"ERROR:\t({__name__}): No config.result_dir"
    filename = config.classification_net + ' ' + config.dataset_class + ' ' + config.time
    filepath = config.result_dir + '/' + filename + '.csv'
    
    content = pd.DataFrame.from_dict(content, orient="index").T
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    if plot:
        results = pd.read_csv(filepath)
        epoch = results["epoch"]
        train_loss = results["train loss"]
        test_loss = results["test acc"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot(epoch, train_loss, color='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(epoch, test_loss, color='tab:blue')
        
        ax1.set_xlabel('Epochs', fontsize=10)
        ax1.set_ylabel('Train Loss', color='tab:red', fontsize=10)
        ax1.grid(alpha=0.4)
        ax2.set_ylabel('Test Acc', color='tab:blue', fontsize=10)
        
        fig.tight_layout()
        plt.savefig(f'{config.result_dir}/{filename}.jpg')
        plt.close()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        
###########################################################################################################
# FUNCTION: classification_trainer()
# 用于分类网络的训练
###########################################################################################################
def classification_trainer(config, net, train_iter, validation_iter, num_epochs, lr):
    # For fast training on GPUs
    if torch.cuda.is_available():
        cudnn.benchmark = True
        
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    net.apply(init_weights)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fuction = nn.CrossEntropyLoss()
    timer = Timer()
    
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            loss = loss_fuction(y_hat, y) # batchmean
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                metric.add(loss*X.shape[0], cal_correct(y_hat, y), X.shape[0])
            timer.stop()
            
        # Show
        print(f'epoch:{epoch+1}')
        if validation_iter is not None:
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            validation_acc = evaluate_accuracy(net, validation_iter)
            print(f'train loss:{train_loss:.3f}, train acc:{train_acc:.3f}, validation acc:{validation_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
                f'on gpu:{config.gpu_id}')
            print(f'--------------------------------------')
            
            # Record Data
            save_results(config, 
                        {
                            "epoch": f"{epoch+1}", 
                            "train loss":f"{train_loss:.5f}", 
                            "train acc":f"{train_acc:.5f}", 
                            "validation acc":f"{validation_acc:0.5f}",
                            "time":f"{timer.sum()}"
                        },
                        plot=True,
                        )
        else:
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            print(f'train loss:{train_loss:.3f}, train acc:{train_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
                f'on gpu:{config.gpu_id}')
            print(f'--------------------------------------')
            
            # Record Data
            save_results(config, 
                        {
                            "epoch": f"{epoch+1}", 
                            "train loss":f"{train_loss:.5f}", 
                            "train acc":f"{train_acc:.5f}", 
                            "time":f"{timer.sum()}"
                        },
                        plot=True,
                        )
            
    # Save Classification_net
    filename = config.classification_net + ' ' + config.dataset_class + ' ' + config.time
    filepath = config.model_save_dir + '/Classification ' + filename + '.pt'
    torch.save(net.state_dict(), filepath)


