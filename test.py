import tqdm
import torch
a = torch.randn(16, 3, 16, 16)
for i in tqdm.tqdm(a):
    pass