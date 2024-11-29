import os
import sys
import random
import cv2
from torchvision import transforms
import torch
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd()) + "/metrics")
from metrics.FeatureNet import ConvNetFeatureExtract
from metrics.mmd import MMD_loss
from metrics.emd import wasserstein_distance
from metrics.swd import sliced_wasserstein_distance
from metrics.fid import compute_fid_kid, make_custom_stats, remove_custom_stats
from metrics.is_ import compute_is


def read_images(images_path, batch_size=50):
    """ real all images in images_path
        return Torch.Tensor as [N, C, H, W] which normalize in [0, 1]."""
    images = []
    images_list = random.sample(os.listdir(images_path), k=batch_size)
    for i, img in enumerate(images_list):
        img_path = images_path + '/' + img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.  # ToTensor
        # images[i] = img
        images.append(img)
    images = np.array(images)
    print(images.shape)
    return torch.FloatTensor(images)

expriment_name = "/GAN_G5e-1_FMLattn_HYPTripw10M05MLPProv2"
# --------------------------------
# FeatureNet metrics
# --------------------------------
def FeatureNetMetrics():
    convnet_feature_extract = ConvNetFeatureExtract(model="inception_v3", workers=4)

    root = "../work_dir_pcb/generator_50_num1500/"
    gan_type = [root + gt for gt in os.listdir(root) if os.path.isdir(root + gt)]
    for gt in gan_type:  # '../work_dir/generator/dragan'
        for cls_ in os.listdir(gt):  # Cr
            epoch = 10000  # maybe final epoch!=10000, change it !!!
            final_epoch = gt + '/' + cls_ + expriment_name + '/epoch%d' % epoch
            real_path = "../dataset/SDPCB/PCB-50-r64/train/" + cls_
            fake_path = final_epoch

            # extract feature
            real_feature = convnet_feature_extract.extractFeature(real_path)
            fake_feature = convnet_feature_extract.extractFeature(fake_path)

            # calculate feature metrics
            mmd = MMD_loss("rbf")(real_feature, fake_feature)  # MMD
            emd = wasserstein_distance(real_feature, fake_feature)  # EMD
            swd = sliced_wasserstein_distance(real_feature, fake_feature)  # SWD

            # write log
            log_path = gt + '/' + cls_ + expriment_name + "/FeatureNet_metrics.txt"
            with open(log_path, "a+") as log:
                item = "Gan_type:%s %s Epoch:%d MMD:%f EMD:%f SWD:%f" % \
                    (gt.split('/')[-1], cls_, epoch, mmd, emd, swd)
                log.write(item + '\n')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# FeatureNetMetrics()


# --------------------------------
# Image metrics
# --------------------------------
def ImageMetrics():
    root = "../work_dir_pcb/generator_50_num1500/"
    gan_type = [root + gt for gt in os.listdir(root) if os.path.isdir(root + gt)]
    for gt in gan_type:  # '../work_dir/generator/dragan'
        for cls_ in os.listdir(gt):  # Cr
            epoch = 9600  # maybe final epoch!=10000, change it !!!
            final_epoch = gt + '/' + cls_ + expriment_name + '/epoch%d' % epoch
            real_path = "../dataset/SDPCB/PCB-50-r64/train/" + cls_
            fake_path = final_epoch

            # real images
            real_images = read_images(real_path)
            fake_images = read_images(fake_path)

            # calculate images metrics
            # 1.EMD
            emd = wasserstein_distance(real_images, fake_images)
            # 2.FID
            custom_name = gt.split('/')[-1] + '_' + cls_.lower()
            remove_custom_stats(custom_name)
            make_custom_stats(custom_name, real_fdir=real_path, mode="clean")
            fid, kid_10e3 = compute_fid_kid(fake_fdir=fake_path, custom_name=custom_name)
            # remove_custom_stats(custom_name)
            # 3.IS
            is_ = compute_is(path=fake_path)
            # 4.JSD ValueError: n_samples=50 should be >= n_clusters=100.
            # change real images as NEU-300-r64, 300 > 100
            # epoch10000:JS=0.326 epoch400:JS=0.279
            # 5.MMD
            MMDLOSS = MMD_loss("rbf")
            X = real_images.reshape(real_images.shape[0], -1)
            Y = fake_images.reshape(fake_images.shape[0], -1)
            mmd_ = MMDLOSS(X, Y).item()  # tensorFloat -> float
            # 6.SWD
            swd_ = sliced_wasserstein_distance(real_images, fake_images)
            swd_ = swd_.cpu().item()  # tensor.cudaFloat -> tensorFloat -> float

            # write log
            log_path = gt + '/' + cls_ + expriment_name + "/Image_metrics.txt"
            # if os.path.exists(log_path):
            #     os.remove(log_path)
            with open(log_path, "a+") as log:
                item = "Gan_type:%s %s Epoch:%d EMD:%.4f FID:%.4f KID_10e3:%.4f IS:%.4f MMD:%.4f SWD:%.4f" % \
                    (gt.split('/')[-1], cls_, epoch, emd, fid, kid_10e3, is_, mmd_, swd_)
                log.write(item + '\n')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ImageMetrics()