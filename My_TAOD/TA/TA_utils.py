import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def init_random_seed(manual_seed):
    """Init the random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print(f'Use random seed:{seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


