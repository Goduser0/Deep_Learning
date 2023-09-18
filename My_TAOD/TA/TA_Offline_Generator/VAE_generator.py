import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import img_255to1, img_1to255
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_VAE import VAE

def TA_VAE_generator():
    model = VAE(3, 128)
    # 选择模型参数和样本图像
    # model.load_state_dict(torch.load('TrainedModels/model_vae_mnist.pth'))
    # dir = "/home/zhouquan/MyDoc/Deep_Learning/mnist_3.jpg"
    
    model.load_state_dict(torch.load('vae_pcb.pth'))
    dir = "./My_Datasets/Classification/PCB-200/Mouse_bite/000939_0_02_02787_13116.bmp"
    
    raw_sample_image = Image.open(dir).convert("RGB")
    raw_sample_image = np.array(raw_sample_image)
    plt_raw = raw_sample_image
    raw_sample_image = img_255to1(raw_sample_image)
    trans = T.Compose([T.ToTensor(), T.Resize((128, 128))])
    sample_image = trans(raw_sample_image)
    sample_image = sample_image.unsqueeze(0)
    
    generated_image = model(sample_image)[0]
    # generated_image = model.sample(1, "cpu")
    generated_image = generated_image.detach().numpy()[0]
    plt_gen = generated_image
    # plt_gen = img_1to255(generated_image)
    # print(np.max(plt_gen), np.min(plt_gen))
    plt_gen = np.transpose(plt_gen, (1, 2, 0))
    # 显示原始图像和生成的图像
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(plt_raw)
    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    plt.imshow(plt_gen)
    plt.savefig("Model_generator.jpg")
    plt.close()


if __name__ == "__main__":
    TA_VAE_generator()
    