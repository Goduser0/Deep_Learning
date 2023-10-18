# generate.py 
import torch
import matplotlib.pyplot as plt
from demo_VAE import model, input_size, mnist # 从 VAE.py 中导入模型、输入大小和 MNIST 数据集

# 加载已训练好的模型
model.load_state_dict(torch.load('vae_minst.pth'))

# 选择mnist的样本图像 
sample_image = mnist[11][0]

# 使用 VAE 的编码器将样本图像编码为 latent variables
mu, log_var = model.encoder(sample_image.view(-1, input_size))

# 将生成的 latent variables 作为输入传递给 VAE 的解码器，生成数字图像
generated_image = model.decoder(mu).view(28, 28)

# 显示原始图像和生成的图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(sample_image.view(28, 28), cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Generated Image')
plt.imshow(generated_image.detach().numpy(), cmap='gray')
plt.savefig('demo_vae_gen.jpg')
plt.close()
