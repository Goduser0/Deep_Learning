import torch
import torchvision.transforms as T
import numpy as np
import imgaug.augmenters as imgaa
import matplotlib.pyplot as plt

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255

######################################################################################################
#### Class:ImgAugmentation
######################################################################################################
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
            imgs (B-C-H-W, torch.uint8)
        Returns:
            imgs (B-C-H-W, torch.uint8)
        """
        if imgs.dtype != torch.uint8:
            raise TypeError
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = np.array(imgs)
        imgs_aug = self.aug_real.augment_images(imgs)
        imgs_aug = torch.from_numpy(imgs_aug)
        imgs_aug = imgs_aug.permute(0, 3, 1, 2)
        return imgs_aug
    
    def imgFakeAug(self, imgs):
        """_summary_
        Args:
            imgs (B-C-H-W, torch.uint8)
        Returns:
            imgs (B-C-H-W, torch.uint8)
        """
        if imgs.dtype != torch.uint8:
            raise TypeError
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = np.array(imgs)
        imgs_aug = self.aug_fake.augment_images(imgs)
        imgs_aug = torch.from_numpy(imgs_aug)
        imgs_aug = imgs_aug.permute(0, 3, 1, 2)
        return imgs_aug

######################################################################################################
#### Test
######################################################################################################
def test():
    trans = T.Compose(
        [
            T.ToTensor(), 
            T.Resize((128, 128)), # (0, 255)
        ]
    )
    data_iter_loader = get_loader('PCB_200', 
                              "./My_TAOD/dataset/PCB_200/0.7-shot/train/1.csv", 
                              32, 
                              4, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    imgAug = ImgAugmentation()
    
    for i, data in enumerate(data_iter_loader):
        raw_img = img_1to255(data[0])
        real_aug = imgAug.imgRealAug(raw_img)
        fake_aug = imgAug.imgFakeAug(raw_img)
        
        plt_raw_img = np.transpose(raw_img.numpy(), (0, 2, 3, 1))
        plt_real_aug = np.transpose(real_aug.numpy(), (0, 2, 3, 1))
        plt_fake_aug = np.transpose(fake_aug.numpy(), (0, 2, 3, 1))
       
        plt.subplot(1, 3, 1)
        plt.title('O')
        plt.imshow(plt_raw_img[0])
        plt.subplot(1, 3, 2)
        plt.title('real aug')
        plt.imshow(plt_real_aug[0])
        plt.subplot(1, 3, 3)
        plt.title('fake aug')
        plt.imshow(plt_fake_aug[0])
        plt.savefig("AUG.jpg")
        plt.close()
        break
    
if __name__ == "__main__":
    test()
    