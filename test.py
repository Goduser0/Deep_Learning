import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model

def calculate_fid(real_images, fake_images):
    inception = InceptionV3(include_top=False, pooling="avg")
    real_images = preprocess_input(real_images)
    fake_images = preprocess_input(fake_images)
    real_features = inception.predict(real_images)
    fake_features = inception.predict(fake_images)
    real_mean, real_cov = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    fake_mean, fake_cov = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    sqrt_trace_cov = sqrtm(np.dot(real_cov, fake_cov))
    fid = np.linalg.norm(real_mean - fake_mean) + np.trace(real_cov + fake_cov - 2 * sqrt_trace_cov)
    return fid

if __name__ == "__main__":
    real_images = np.random.rand(100, 299, 299, 3)
    fake_images = np.random.rand(100, 299, 299, 3)
    fid = calculate_fid(real_images, fake_images)
    print(f"FID: {fid}")