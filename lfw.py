import albumentations as alb
from albumentations.pytorch import ToTensorV2

import copy
import random
import torch
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from pathlib import Path
from torch.utils.data import Dataset



class LFWDataset(Dataset):
    def __init__(self, data_folder, image_size=160, data_slice=None, transform=None, verbose=False):
        paths = Path('~/datasets/lfw/lfw_mtcnnpy_160').expanduser().glob('**/*_*.png')
        self.files = [str(p) for p in paths]

        if data_slice:
            assert len(data_slice) == 2 and 0 <= data_slice[0] and data_slice[0] <= data_slice[1] and data_slice[1] <= 1
            self.files = self.files[int(data_slice[0]*len(self.files)):int(data_slice[1]*len(self.files))]

        self.image_size = image_size

        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise NotImplementedError()

        sample_path = self.files[idx]
        image = imread(sample_path)
        image = cvtColor(image, COLOR_BGR2RGB)

        if image.shape[:2] != (self.image_size, self.image_size):
            raise RuntimeError(f'{sample_path} has wrong shape: {image.shape}')

        label_str = sample_path[-8]
        assert label_str in '01'

        if self.transform:
            image = self.transform(image=image)['image']

        if self.verbose:
            print(f'Loaded {sample_path}')
        return (image, int(label_str))


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = alb.Compose([t for t in dataset.transform if not isinstance(t, (alb.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from torchvision import transforms
    from torchvision.transforms import ToTensor
    from matplotlib import pyplot as plt

    transform = alb.Compose([
                                    alb.Resize(64, 64),
                                    alb.HorizontalFlip(p=0.5),
                                    alb.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                                    alb.RandomBrightnessContrast(p=0.5),
                                    alb.GaussNoise(var_limit=(2.0, 5.0), mean=0, always_apply=False, p=0.3),
                                    alb.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=False, p=0.3),
                                    alb.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=50, val_shift_limit=30, always_apply=False, p=0.5),
                                    alb.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
                                    # alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
                                    alb.Normalize(),
                                    ToTensorV2(),
                                    ])

    lfw = LFWDataset(data_folder='darusik/datasets/lfw/lfw_mtcnnpy_160', image_size=160, transform=transform, verbose=True)

    for i in range (40):
        image, label = lfw[i]
        print(image, label)



        # plt.imshow(image)
        # plt.show()
    # np_image = np.transpose(image.numpy(), (2, 1, 0))
    # print(np_image.min(), np_image.max())
    # plt.imshow(np_image)
    # plt.show()

    random.seed(82)
    visualize_augmentations(lfw)