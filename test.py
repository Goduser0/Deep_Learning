import os
import torch
import torch.nn as nn
from torch.autograd import Variable

i = torch.randint(1, 50, (1, 4))
print(f"{i}:{i.dtype}")

b = Variable(i)
print(f"{b}:{b.dtype}")