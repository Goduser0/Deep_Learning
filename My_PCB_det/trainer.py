import datetime
import os
import argparse
import warnings
import time
import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.FasterRCNN_train import FasterRCNN
from net.Suggestion_box import FasterRCNNTrainer, FasterRCNNTrainer_fpn, get_lr_scheduler, set_optimizer_lr, weights_init

from dataloader import FRCNNDataset, frcnn_dataset_collate
from utils import get_classes, logger
from utils_fit import fit_one_epoch
from callbacks import LossHistory, EvalCallback

########################################################################################################
#### Config
########################################################################################################
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument("--train_gpu", type=list, default=[1,])
parser.add_argument("--fp16", type=bool, default=True)

parser.add_argument("--classes_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集/cls_classes.txt') # Imagenet pretrained
parser.add_argument("--model_path", type=str, default='My_PCB_det/Weights/voc_weights_resnet.pth')

parser.add_argument("--input_shape", type=int, default=600)
parser.add_argument("--backbone", type=str, default='resnet50', choices=['vgg', 'resnet50', 'resnet101', 'resnet50_FPN'])
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--anchors_size", type=list, default=[4, 16, 32, 64, 128])

parser.add_argument("--Init_Epoch", type=int, default=0)
parser.add_argument("--UnFreeze_Epoch", type=int, default=200)
parser.add_argument("--Unfreeze_batch_size", type=int, default=15)
parser.add_argument("--Freeze_Train", type=bool, default=False)

parser.add_argument("--Init_lr", type=float, default=1e-4)
parser.add_argument("--Min_lr", type=float, default=1e-4 * 0.01)
parser.add_argument("--optimizer_type", type=str, default='adam')
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--lr_decay_type", type=str, default='step')
parser.add_argument("--save_period", type=int, default=50)

parser.add_argument("--save_dir", type=str, default='My_PCB_det/logs')

parser.add_argument("--eval_flag", type=bool, default=False)
parser.add_argument("--eval_period", type=int, default=50)

parser.add_argument("--num_workers", type=int, default=12)

parser.add_argument("--train_annotation_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集_VOC/trainval.txt')
parser.add_argument("--test_annotation_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集_VOC/test.txt')

parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

config = parser.parse_args()
logger(config)
os.makedirs(os.path.join(config.save_dir, config.time))
config.save_dir = os.path.join(config.save_dir, config.time)

class_names, num_classes = get_classes(config.classes_path)
input_shape = [config.input_shape, config.input_shape]

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in config.train_gpu)
n_gpus_per_node = len(config.train_gpu)
# print('Number of devices: {}'.format(n_gpus_per_node))

model = FasterRCNN(num_classes, anchor_scales=config.anchors_size, backbone=config.backbone, pretrained=config.pretrained)

if not config.pretrained:
    weights_init(model)

if config.model_path != '':
    print('Load weight from {}.'.format(config.model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(config.model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    
    print("\nSuccess Load Key:", str(load_key)[:500], "...\nSuccess Load Key Num:", len(load_key))
    print("\nFail Load Key:", str(no_load_key)[:500], "...\nFail Load Key Num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示,head部分没有载入是正常现象,Backbone部分没有载入是错误的。\033[0m")

time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
log_dir = os.path.join(config.save_dir, "loss_"+str(time_str))
loss_history = LossHistory(log_dir, model, input_shape)

if config.fp16:
    from torch.cuda.amp import GradScaler as GradScaler
    scaler = GradScaler()
else:
    scaler = None

model_train = model.train()
model_train = torch.nn.DataParallel(model_train)
cudnn.benchmark = True
model_train = model_train.cuda()

with open(config.train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(config.test_annotation_path, encoding='utf-8') as f:
    val_lines = f.readlines()
num_train  = len(train_lines)
num_val = len(val_lines)
print(f"train:{num_train}, val:{num_val}")

wanted_step = 5e4 if config.optimizer_type == "sgd" else 1.5e4
total_step  = num_train // config.Unfreeze_batch_size * config.UnFreeze_Epoch
if total_step <= wanted_step:
    if num_train // config.Unfreeze_batch_size == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
    wanted_epoch = wanted_step // (num_train // config.Unfreeze_batch_size) + 1
    print("\n\033[1;33;44m[Warning] 使用%s优化器时,建议将训练总步长设置到%d以上。\033[0m"%(config.optimizer_type, wanted_step))
    print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d,Unfreeze_batch_size为%d,共训练%d个Epoch,计算出总训练步长为%d。\033[0m"%(num_train, config.Unfreeze_batch_size, config.UnFreeze_Epoch, total_step))
    print("\033[1;33;44m[Warning] 由于总训练步长为%d,小于建议总步长%d,建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

if True:
    UnFreeze_flag = False
    model.freeze_bn()
    batch_size = config.Freeze_batch_size if config.Freeze_Train else config.Unfreeze_batch_size
    nbs = 16
    lr_limit_max = 1e-4 if config.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if config.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (config.momentum, 0.999), weight_decay = config.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = config.momentum, nesterov=True, weight_decay = config.weight_decay)
        }[config.optimizer_type]
    
    lr_scheduler_func = get_lr_scheduler(config.lr_decay_type, Init_lr_fit, Min_lr_fit, config.UnFreeze_Epoch)
    
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    # if epoch_step == 0 or epoch_step_val == 0:
    #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    
    train_dataset = FRCNNDataset(train_lines, input_shape, train = True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train = False)
    
    gen = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = config.num_workers, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = config.num_workers, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
    
    if config.backbone == "resnet50_FPN":
        train_util = FasterRCNNTrainer_fpn(model_train, optimizer)
    else:
        train_util = FasterRCNNTrainer(model_train, optimizer)
        
    eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, True, eval_flag=config.eval_flag, period=config.eval_period)
    
    for epoch in range(config.Init_Epoch+1, config.UnFreeze_Epoch+1):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, config.UnFreeze_Epoch, True, config.fp16, scaler, config.save_period, config.save_dir)
    
    loss_history.writer.close()