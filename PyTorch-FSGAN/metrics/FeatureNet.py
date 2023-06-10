# ----------------------------------------------
# this code create a Convd Feature Extract Net
# ----------------------------------------------
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from scipy import linalg
import random


def read_images(images_path, transform=None, batch_size=30):
    """ real all images in images_path
        return Torch.Tensor as [N, C, H, W] which normalize in [0, 1].
        Add Transform function. *args : trans. Don't use Batch.
        the GANDataset and DataLoader must return a dataset with batch..."""
    images = []
    images_list = random.sample(os.listdir(images_path), k=batch_size)
    for i, img in enumerate(images_list):
        img_path = images_path + '/' + img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # trans
        if transform:
            img = transforms.ToPILImage()(img)
            img = transform(img)

        images.append(img.numpy())  # add ndarray type
    images = np.array(images)
    # print(images.shape)
    return torch.FloatTensor(images)  # Return torch.Tensor


class ConvNetFeatureExtract(object):
    def __init__(self, model='resnet34', workers=4):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.workers = workers
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def extractFeature(self, images_path):
        # build images dataset
        images = read_images(images_path, transform=self.trans)
        print("Extracting Features...")
        with torch.no_grad():
            input = images.cuda()
            if self.model == 'vgg' or self.model == 'vgg16':
                fconv = self.vgg.features(input).view(input.size(0), -1)
                print(self.model + " feature shape:", fconv.shape)
            elif self.model.find('resnet') >= 0:
                fconv = self.resnet_feature(input).mean(3).mean(2).squeeze()
                print(self.model + " feature shape:", fconv.shape)
            elif self.model == 'inception' or self.model == 'inception_v3':
                fconv = self.inception_feature(input).mean(3).mean(2).squeeze()
                print(self.model + " feature shape:", fconv.shape)
            else:
                raise NotImplementedError
            feature_conv = fconv.data.cpu()
        return feature_conv


# --------------------
# metrics
# --------------------
def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from knn import compute_score
    from mmd import MMD_loss
    from emd import wasserstein_distance
    from swd import sliced_wasserstein_distance, slice_wasserstein_distance

    # real_path = "../dataset/NEU/NEU-300-r64/Cr"
    # fake_path = "../work_dir/generator/wgan-gp/Cr"
    real_path = "../dataset/NEU/NEU-50-r64/train/Sc"
    fake_path = "../work_dir/samples/random_samples/Sc"

    convnet_feature_extract = ConvNetFeatureExtract(model="inception_v3", workers=4)
    real_feature = convnet_feature_extract.extractFeature(real_path)
    fake_feature = convnet_feature_extract.extractFeature(fake_path)

    # --------------------------------
    # calculate feature metrics
    # --------------------------------
    # compute_score(real_feature, fake_feature)
    # print(MMD_loss("rbf")(real_feature, fake_feature))
    # print(wasserstein_distance(real_feature, fake_feature))
    # print(fid(real_feature, fake_feature))
    print(sliced_wasserstein_distance(real_feature, fake_feature))
    # print(slice_wasserstein_distance(real_feature, fake_feature))  # assert image1.ndim == 4 and image2.ndim == 4, must be images as [N, C, H, W]
