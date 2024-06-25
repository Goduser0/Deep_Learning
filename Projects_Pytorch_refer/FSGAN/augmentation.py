from torch import from_numpy
from numpy import array
import imgaug.augmenters as iaa


class ImgAugmentation():
    """Image Data Augmentation
    `Aim`:
        Augmentation both real imgs and fake imgs with SAME Random State
        Different aug method at different loader
    """
    def __init__(self):
        self.seed = 42
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=self.seed)  # define `sometimes` handle p=0.5
        self.aug_real = iaa.Sequential([
            # A seq to augmentation, random choose 1-5 aug ways
            iaa.SomeOf((1, 5),
                       [
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           self.sometimes(iaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                               iaa.Fliplr(random_state=self.seed),
                               iaa.Flipud(random_state=self.seed)
                           ], random_state=self.seed),
                           iaa.Crop(keep_size=True, random_state=self.seed)
                       ], random_order=True, random_state=self.seed)
        ], random_order=True, random_state=self.seed)  # handle oder same in one batch, diff in diff batch
        self.aug_fake = iaa.Sequential([
            # A seq to augmentation, random choose 1-5 aug ways
            iaa.SomeOf((1, 5),
                       [
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           iaa.contrast.LinearContrast(random_state=self.seed),
                           self.sometimes(iaa.Affine(rotate=(-20, 20), mode="symmetric", random_state=self.seed)),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(scale=(0.0, 0.01 * 255), random_state=self.seed),
                               iaa.Fliplr(random_state=self.seed),
                               iaa.Flipud(random_state=self.seed)
                           ], random_state=self.seed),
                           iaa.Crop(keep_size=True, random_state=self.seed)
                       ], random_order=True, random_state=self.seed)
        ], random_order=True, random_state=self.seed)  # handle oder same in one batch, diff in diff batch

    def imgRealAugment(self, imgs):
        """Returns:
            torch.int8: torch.from_numpy()
        """
        imgs = array(imgs)
        imgs_aug = self.aug_real.augment_images(imgs) # replaced seq(images=imgs), return ndarray
        return from_numpy(imgs_aug)

    def imgFakeAugment(self, imgs):
        """Returns:
            torch.int8: torch.from_numpy()
        """
        imgs = array(imgs)
        imgs_aug = self.aug_fake.augment_images(imgs)
        return from_numpy(imgs_aug)
