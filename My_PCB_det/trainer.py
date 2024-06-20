import datetime
import os
import argparse
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader



from net.FasterRCNN_train import FasterRCNN
from net.Suggestion_box import FasterRCNNTrainer, FasterRCNNTrainer_fpn, get_lr_scheduler, set_optimizer_lr, weights_init
from utils import get_classes
from callbacks import LossHistory

########################################################################################################
#### Config
########################################################################################################
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument("--train_gpu", type=list, default=[0,])
parser.add_argument("--fp16", type=bool, default=True)

parser.add_argument("--classes_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集/cls_classes.txt')
parser.add_argument("--model_path", type=str, default='')

parser.add_argument("--input_shape", type=list, default=[600, 600])
parser.add_argument("--backbone", type=str, default='resnet50_FPN', choices=['vgg', 'resnet50', 'resnet101', 'resnet50_FPN'])
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--anchors_size", type=list, default=[4, 16, 32, 64, 128])

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

parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--train_annotation_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集/Mouse_bite_txt/01_Mouse_bite.txt')
parser.add_argument("--test_annotation_path", type=str, default='My_Datasets/Detection/PCB_瑕疵初赛样例集/Mouse_bite_txt/01_Mouse_bite.txt')

config = parser.parse_args()
print(config)  

class_names, num_classes = get_classes(config.classes_path)

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
loss_history = LossHistory(log_dir, model, input_shape=config.input_shape)

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
    val_lines   = f.readlines()
num_train  = len(train_lines)
num_val = len(val_lines)