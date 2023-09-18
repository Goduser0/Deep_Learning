import torch
import torchvision.transforms as T
import numpy as np
import imgaug.augmenters as imgaa
import matplotlib.pyplot as plt

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255

class ImgAugmentation():
    """Image Data Augmentation
    `Aim`:
        Augmentation both real imgs and fake imgs with SAME Random State
        Different aug method at different loader
    """
    def __init__(self):
        self.seed = 42
        self.sometimes = lambda aug: imgaa.Sometimes(0.5, aug, random_state=self.seed)
        
        self.aug_real = imgaa.Sequential(
            [
                imgaa.SomeOf(
                    (1, 5),
                    [
                        imgaa.contrast.LinearContrast(random_state=self.seed),
                        imgaa.contrast.LinearContrast(random_state=self.seed),
                        self.sometimes(imgaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                        imgaa.OneOf(
                            [
                                imgaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                                imgaa.Fliplr(random_state=self.seed),
                                imgaa.Flipud(random_state=self.seed),
                            ],
                            random_state=self.seed,
                        ),
                        imgaa.Crop(keep_size=True, random_state=self.seed),
                    ],
                    random_order=True,
                    random_state=self.seed         
                )
            ], 
            random_order=True, 
            random_state=self.seed
        )
        
        self.aug_fake = imgaa.Sequential(
            [
                imgaa.SomeOf(
                    (1, 5),
                    [
                        imgaa.contrast.LinearContrast(random_state=self.seed),
                        imgaa.contrast.LinearContrast(random_state=self.seed),
                        self.sometimes(imgaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                        imgaa.OneOf(
                            [
                                imgaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                                imgaa.Fliplr(random_state=self.seed),
                                imgaa.Flipud(random_state=self.seed),
                            ],
                            random_state=self.seed,
                            ),
                        imgaa.Crop(keep_size=True, random_state=self.seed)
                    ],
                    random_order=True,
                    random_state=self.seed,
                )    
            ],
            random_order=True,
            random_state=self.seed,
        )
        
    def imgRealAug(self, imgs):
        """_summary_
        Args:
            imgs (_type_)
        Returns:
            torch.Tensor
        """
        imgs = np.array(imgs)
        imgs_aug = self.aug_real.augment_images(imgs)
        return torch.from_numpy(imgs_aug)
    
    def imgFakeAug(self, imgs):
        """_summary_
        Args:
            imgs (_type_)
        Returns:
            torch.Tensor
        """
        imgs = np.array(imgs)
        imgs_aug = self.aug_fake.augment_image(imgs)
        return torch.from_numpy(imgs_aug)

if __name__ == "__main__":
    trans = T.Compose(
        [
            T.ToTensor(), 
            T.Resize((128, 128)), # (0, 255)
        ]
    )
    data_iter_loader = get_loader('PCB_200', 
                              "./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv", 
                              32, 
                              4, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    imgAug = ImgAugmentation()
    
    for i, data in enumerate(data_iter_loader):
        raw_img = data[0] # [B, C, H, W] -1~1
        real_imgs_aug = img_1to255(raw_img.numpy())
        real_imgs_aug = imgAug.imgRealAug(real_imgs_aug)
        
        raw_img = img_1to255(raw_img[0].numpy())
        real_img_aug = img_1to255(real_imgs_aug[0].numpy())
        raw_img = np.transpose(raw_img, (1, 2, 0))
        real_img_aug = np.transpose(real_img_aug, (1, 2, 0))
        
        plt.subplot(1, 2, 1)
        plt.title('O Image')
        plt.imshow(raw_img)
        plt.subplot(1, 2, 2)
        plt.title('G Image')
        plt.imshow(real_img_aug)
        plt.savefig("AUG.jpg")
        plt.close()
        
        break
    