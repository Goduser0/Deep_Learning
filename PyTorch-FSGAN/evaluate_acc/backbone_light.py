import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from timm import create_model
from timm.data.transforms_factory import create_transform
from timm.data import ImageDataset
from timm.optim import create_optimizer_v2
from timm.scheduler import StepLRScheduler

import os

parse = argparse.ArgumentParser()
parse.add_argument("--gpu_ids", type=int, default=1)
parse.add_argument("--model_name", type=str, default="vgg16", choices=["vgg16", "mobilenetv3_large_100", "efficientnet_b0"])
parse.add_argument("--pretrained", type=bool, default=False)
parse.add_argument("--num_classes", type=int, default=6)
parse.add_argument("--train_path", type=str, default="../dataset/NEU/NEU-50-r64/train")
parse.add_argument("--test_path", type=str, default="../dataset/NEU/NEU-50-r64/test")
parse.add_argument("--n_epochs", type=int, default=10)
parse.add_argument("--batch_size", type=int, default=32)
parse.add_argument("--log_name", type=str, default="log")
parse.add_argument("--output", type=str, default="results")
config = parse.parse_args()
print(config)


os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids)
model_name = config.model_name


model = create_model(model_name, pretrained=config.pretrained, num_classes=config.num_classes)
# net = nn.DataParallel(model).cuda()  # multi-GPU load and save, DDP
net = model.cuda()


def write_log(log_path, item):
    with open(log_path, "a+") as log:
        log.write(item + '\n')
os.makedirs(config.output + "/" + model_name, exist_ok=True)
log_path = config.output + "/" + model_name + "/" + config.log_name + '.txt'


def create_dataloader_iterator():
    trans = create_transform(224, interpolation="bicubic")
    train_dataset = ImageDataset(config.train_path, transform=trans)
    test_dataset = ImageDataset(config.test_path, transform=trans)
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=1)
    test_dl  = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1)
    return train_dl, test_dl, trans

def create_optimizer(t_init=30):
    optimizer = create_optimizer_v2(net.parameters(), opt="sgd", lr=0.001)
    scheduler = StepLRScheduler(optimizer, warmup_lr_init=0.0001, warmup_t=3, decay_t=10)
    return optimizer, scheduler

def creat_loss():
    loss = CrossEntropyLoss()
    return loss

def main():
    # load dataset
    train_dl, test_dl, trans = create_dataloader_iterator()
    optimizer, scheduler = create_optimizer()
    loss_fn = creat_loss()
    print(trans)
    write_log(log_path, str(config) + '\n' + str(trans))

    # epochs, just test
    train_epochs = config.n_epochs
    num_epochs = train_epochs

    # train
    net.train()
    for epoch in range(num_epochs):
        print("Epoch {}\n-------------------------------".format(epoch + 1))
        train_step = 0
        num_steps_per_epoch = len(train_dl)
        # num_updates = epoch * num_steps_per_epoch

        for batch in train_dl:
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step_update(num_updates=num_updates)

            train_step += 1
            if train_step % 5 == 0:
                print("step: {} loss: {}".format(train_step, loss.item()))

        # scheduler.step(epoch + 1)

        # evaluate
        net.eval()
        num_batchs = len(test_dl)
        size = len(test_dl.dataset)
        # print(num_batchs, size)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs_test, targets_test in test_dl:
                inputs_test, targets_test = inputs_test.cuda(), targets_test.cuda()
                pred = net(inputs_test)

                test_loss += loss_fn(pred, targets_test).item()
                correct   += (pred.argmax(1) == targets_test).type(torch.float).sum().item()
        test_loss /= num_batchs
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        item = "Epoch: {}".format(epoch + 1) + '\n' + '-'*30 + '\n' + \
            "Accuracy: {:.2f}, Avg loss: {:.6f}".format(correct * 100, test_loss)
        write_log(log_path, item)


if __name__ == '__main__':
    main()
    # torch.save(net, "../work_dir/_base_/" + model_name + ".pth")  # 多卡保存
