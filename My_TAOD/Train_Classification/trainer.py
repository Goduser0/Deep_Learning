import os
import time
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.backends import cudnn


###########################################################################################################
# 通用函数
###########################################################################################################
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

def validation_accuracy(net, validation_iter):
    """计算模型在验证集上的性能"""
    if isinstance(net, nn.Module):
        net.eval()
    
    with torch.no_grad():
        y_list = []
        y_hat_list = [] 
        for X, y in validation_iter:
            if isinstance(X, list):
                X = [x.cuda() for x in X]
            else:
                X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            
            y_list += list(y.cpu().numpy())
            y_hat = y_hat.argmax(dim=1)
            y_hat_list += list(y_hat.cpu().numpy())
            
    return accuracy_score(y_list, y_hat_list)



def classification_record_data(config, save_dir, content, flag_plot=False):
    """将实验数据存为csv文件"""
    filepath = save_dir + '/data.csv'
    content = pd.DataFrame.from_dict(content, orient="index").T
    # 写入新数据
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    # 读取文件所有数据后绘图
    if flag_plot:
        results = pd.read_csv(filepath)
        
        epoch = results["epoch"]
        num_epochs = results["num_epochs"]
        batch = results["batch"]
        num_batchs = results["num_batchs"]
        
        train_loss = results["train_loss"]
        train_acc = results["train_acc"]
        if "validation_acc" in results:
            validation_acc = results["validation_acc"]
        train_speed = results["train_speed"]
        time = results["time"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot(epoch, train_acc, label="train_acc", color='tab:red')
        if "validation_acc" in results:
            ax1.plot(epoch, validation_acc, label="validation_acc", color='tab:blue')
        ax1.set_xlabel('Epochs', fontsize=10)
        ax1.set_ylabel('Acc', color='tab:red', fontsize=10)
        ax1.grid(alpha=0.4)
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{save_dir}/train_Acc.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot(epoch, train_loss, label="train_loss")
        fig.tight_layout()
        plt.savefig(f'{save_dir}/train_Loss.jpg')
        plt.close()
        
# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
#         nn.init.xavier_uniform_(m.weight)
        
###########################################################################################################
# FUNCTION: classification_trainer()
###########################################################################################################
def classification_trainer(config, save_dir, net, train_iter, validation_iter, num_epochs, lr):
    # For fast training on GPUs
    if torch.cuda.is_available():
        cudnn.benchmark = True
        
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    # net.apply(init_weights)
    net.cuda()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fuction = nn.CrossEntropyLoss()
    timer = Timer()
    
    model_save_path = save_dir + '/models'
    os.makedirs(model_save_path, exist_ok=False)
    
    for epoch in tqdm.trange(1, num_epochs+1, desc=f"On training"):
        loss_list = []
        y_list = []
        y_hat_list = []
        net.train()
        for batch_idx, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            loss = loss_fuction(y_hat, y) # batchmean
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                loss_list.append(loss*X.shape[0])
                y_list += list(y.cpu().numpy())
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat.argmax(dim=1)
                y_hat_list += list(y_hat.cpu().numpy())
            timer.stop()
            
        # Show
        if validation_iter is not None:
            train_loss = sum(loss_list) / len(y_list)
            train_acc = accuracy_score(y_list, y_hat_list)
            validation_acc = validation_accuracy(net, validation_iter)
            train_speed = len(y_list) * num_epochs / timer.sum()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Train loss: %.3f] [Train acc: %.3f] [Validation acc: %.3f] [%.1f examples/sec] [On GPU:%s]"
                %
                (epoch, num_epochs, batch_idx+1, len(train_iter), train_loss, train_acc, validation_acc, train_speed, config.gpu_id)
            )
            # Record Data
            classification_record_data(config, save_dir, 
                        {
                            "epoch": f"{epoch}",
                            "num_epochs": f"{num_epochs}",
                            "batch": f"{batch_idx+1}",
                            "num_batchs": f"{len(train_iter)}",
                            
                            "train_loss":f"{train_loss:.5f}",
                            "train_acc":f"{train_acc:.5f}", 
                            "validation_acc":f"{validation_acc:.5f}",
                            "train_speed":f"{train_speed:.1f}",
                            "time":f"{timer.sum()}"
                        },
                        flag_plot=True,
                        )
        else:
            train_loss = sum(loss_list) / len(y_list)
            train_acc = accuracy_score(y_list, y_hat_list)
            train_speed = len(y_list) * num_epochs / timer.sum()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Train loss: %.3f] [Train acc: %.3f] [%.1f examples/sec] [On GPU:%s]"
                %
                (epoch, num_epochs, batch_idx+1, len(train_iter), train_loss, train_acc, train_speed, config.gpu_id)
            )
            # Record Data
            classification_record_data(config, save_dir, 
                        {
                            "epoch": f"{epoch}",
                            "num_epochs": f"{num_epochs}",
                            "batch": f"{batch_idx+1}",
                            "num_batchs": f"{len(train_iter)}",
                            
                            "train_loss":f"{train_loss:.5f}",
                            "train_acc":f"{train_acc:.5f}", 
                            "train_speed":f"{train_speed:.1f}",
                            "time":f"{timer.sum()}"
                        },
                        flag_plot=True,
                        )
        ##################################################################################
        ## Checkpoint
        ##################################################################################
        #-------------------------------------------------------------------
        # Save Classification_net
        #-------------------------------------------------------------------
        if epoch % (num_epochs // 10) == 0:
            net_path = model_save_path + f"/{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            }, net_path)
        if epoch == num_epochs:
            net_path = model_save_path + f"/{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
            }, net_path)
            
###########################################################################################################
# FUNCTION: svm_trainer()
###########################################################################################################
def svm_trainer(config, net, train_iter, validation_iter, num_epochs, lr):
    pass
    
###########################################################################################################
# Test
###########################################################################################################
def test():
    pass
    y = torch.tensor([1, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])   
    
    with torch.no_grad():
        print(y)
        print(y_hat)
        y_hat = y_hat.argmax(dim=1)
        print(y_hat)
        print(accuracy_score(y, y_hat))

if __name__ == "__main__":
    test()
