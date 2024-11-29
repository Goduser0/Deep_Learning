import numpy as np
import cv2
import os
import imgaug.augmenters as iaa
from pathlib import Path

dataset_name = "PCB-50-r64"
initial_num = 50  # pad NEU-50-r64/train : 50x6cls
padding_num = 30  # multiple, for building few-shot CNN model
category = ['Cs', 'No', 'Ns', 'Op', 'Sh']  # 6 cls name



def get_src_img(str_):
    """get image ndarray according to category.
    :param str_: category
    :return: ndarray
    """
    img_root = Path('../dataset/SDPCB/' + dataset_name + '/train')
    img_root = img_root / str_
    img_list = []
    for image in img_root.glob('*.png'):
        image_path = './' + str(image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
    img_ndarray = np.array(img_list)
    return img_ndarray


def offline_aug(type_):
    # ---------------------------------------
    # 1. geometric transform
    #    |_ rotate, flip
    #    |_ crop, resize
    # 2. color transform
    #    |_ contrast
    #    |_ gaussiannoise
    # 3. geometric-color transform
    #    |_ keep same as augmentation in GAN
    # ---------------------------------------
    # TODO : online_aug
    # ---------------------------------------
    # 1. geometric transform
    # 2. color transform
    # 3. auto augmentation
    # 4. mix transform : mosaic, mixup, cutmix
    # ---------------------------------------

    assert type_ in ["geometric", "color", "geo-color"]
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # define sometimes probility=0.5

    if type_ == "geometric":
        seq = iaa.Sequential([
            iaa.SomeOf((0, 4),
                [
                    iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                    iaa.Fliplr(),
                    iaa.Flipud(),
                    iaa.Crop(keep_size=True)
            ])
        ])

    if type_ == "color":
        seq = iaa.Sequential([
            iaa.SomeOf((0, 3),
                [
                    iaa.contrast.LinearContrast(),
                    iaa.contrast.LinearContrast(),
                    iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255)),
            ])
        ])

    if type_ == "geo-color":
        seq = iaa.Sequential([
            iaa.SomeOf((1, 5),
                [
                    iaa.contrast.LinearContrast(),
                    iaa.contrast.LinearContrast(),
                    sometimes(iaa.Affine(rotate=(-20, 20), mode='symmetric')),
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255)),
                        iaa.Fliplr(),
                        iaa.Flipud()
                    ]),
                    iaa.Crop(keep_size=True)
            ])
        ])
    return seq


def data_padding_cls(str_, type_, seq):  # str_ = category{6}
    """
    pad choose categoryu
    :param str_: category
    :return: pad ndarray
    """
    img_aug_concat = np.ones((1, 64, 64, 3))
    img = get_src_img(str_)  # img.shape = (50, 64, 64, 3)
    for i in range(padding_num):
        img_aug = seq(images=img)
        img_aug_ = img_aug.reshape((initial_num, 64, 64, 3))
        if i == 0:
            img_aug_concat = img_aug_
        else:
            img_aug_concat = np.concatenate((img_aug_concat, img_aug_), axis=0)

    path = '../dataset/SDPCB/' + dataset_name + '-pad/' + type_ + '/train_' + str(initial_num * padding_num) + '/' + str_  # save path
    os.makedirs(path, exist_ok=True)
    for i in range(img_aug_concat.shape[0]):  # shape[0] = initial_num * padding_num
        save_path = os.path.join(path, str(i) + '_pad.png')
        cv2.imwrite(save_path, img_aug_concat[i])
        print('The ' + str(i + 1) + '/' + str(initial_num * padding_num) + ' has been Expanded and Saved')
    return img_aug_concat


def data_padding_all(type_, seq):
    img_all = np.zeros((1, 64, 64, 3))
    for i in range(len(category)):
        print('------------ Now Expanding The category_' + str(i) + ' : ' + str(category[i]) + ' ------------')
        if i == 0:
            img_all = data_padding_cls(category[i], type_, seq)
        else:
            img_all = np.concatenate((img_all, data_padding_cls(category[i], type_, seq)), axis=0)
        pass
    return img_all

if __name__ == '__main__':
    for type_ in ["geometric", "color", "geo-color"]:
        seq = offline_aug(type_)
        data_padding_all(type_, seq)
    pass