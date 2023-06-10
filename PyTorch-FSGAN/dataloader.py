from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os


class GANDataset(Dataset):
    """GAN Dataset with no labels
    Args:
        Dataset (Torch.Dataset): MyDataset
    """
    def __init__(self, images_path, transform=None):
        self.images = [images_path + '/' + img for img in os.listdir(images_path)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        # read image with cv2, return ndarray unit8
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        if self.transform:
            image = self.transform(image)

        return image


def transform(resize, totensor, normalize):
    """Image Transorm
    Args:
        resize : resize, need a PIL image.
        totensor : div(255.) & permute((0, 2, 3, 1)) -> [N, C, H, W]
        normalize : (pixel - mean) / std at 3 channels
    Returns:
        _type_: torch.FloatTensor
    """
    options = []
    if resize:
        # cv2 return is `ndarray`.
        options.append(transforms.ToPILImage())
        options.append(transforms.Resize((64, 64)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    return transform


def dataLoader(images_path, transform=None, batch_size=64, shuffle=True, num_workers=2, drop_last=True):
    """Data Loader from GANDataset
    Returns:
        torch.unit8: Each pixel in the image, without `ToTensor` and `Normalize` for Augmentation at unit8
    """
    dataset = GANDataset(images_path, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return loader
