import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))