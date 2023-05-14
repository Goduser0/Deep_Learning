from torch.utils import data
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image

trans = []
trans.append(T.ToTensor())
trans = T.Compose(trans)
df = pd.read_csv('/home/zhouquan/MyDoc/DL_Learning/elpv.csv')
print(type(df['probs'].unique().tolist()[0]))