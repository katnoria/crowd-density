import logging
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transforms import *


class CCDataset(Dataset):
    """Crowd-Couting dataset Loader"""

    def __init__(self, dirname, transform=None, debug=True, sample_images=None):
        """
        Parameters
        ----------
        dirname: Location of image folder (i.e /path/to/train, /path/to/val)
        transform: List of Transforms (default: None)
        debug: Prints out additional message in debug mode (default: False)
        sample_images: Use the List of images instead of reading from dirname
        """
        self.dirname = dirname
        self.images = glob(f"{self.dirname}/img/*")
        if sample_images is not None:
            self.images = sample_images
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.images[idx]
        image = Image.open(fname)
        image = image.convert('RGB')
        den_name = fname.split("/")[-1].replace(".jpg", ".csv")
        den = np.genfromtxt(f"{self.dirname}/den/{den_name}", delimiter=",").astype(np.float32)

        sample = {'image': image, 'den': den, 'fname': fname}

        if self.transform:
            # if outputs.size()[-1] != den.size()[-1]:
            # import pdb; pdb.set_trace();

            if self.debug:
                logging.debug(f"image:{image.size}: {type(image)}, den:{den.shape}:{type(den)}")

            sample = self.transform(sample)

            if self.debug:
                logging.debug(f"image:{sample['image'].size()}:, den:{sample['den'].shape}:{type(den)}")

        return sample


class CCDataLoader(object):
    """Dataloader Wrapper

    If I see now, this is not really required so we can do away with it
    """

    def __init__(self, dirname, transforms, bs=1, shuffle=True, num_workers=1, sample_images=None):
        """
        Parameters
        ----------
        dirname: Location of image folder (i.e /path/to/train, /path/to/val)
        transform: List of Transforms (default: None)
        bs: Batch size (default: 1)
        shuffle: Bool, Shuffle dataset (default: True)
        num_workers: Number of dataloader workers (default: 1)
        sample_images: Use the List of images instead of reading from dirname
        """
        self.ds = CCDataset(dirname=dirname, transform=transforms, sample_images=sample_images)
        self.dataloader = DataLoader(self.ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.ds)


class CustomEvalDataset(Dataset):
    """Crowd-Couting dataset Loader"""

    def __init__(self, dirname, images=None, transform=None, debug=False):
        self.dirname = dirname
        if images is not None:
            self.images = images
        else:
            self.images = glob(self.dirname)

        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.images[idx]
        image = Image.open(fname)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, fname


def default_train_transforms(output_size=448, factor=4):
    """Training set transforms

    Parameters
    ----------
    output_size: Resize the input image into square(width:output_size, height:output_size)
    factor: Scale down factor to apply on input image (default: 4)
    """
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        RandomFlip(),
        # ScaleDown(factor),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])


def default_val_transforms(output_size=448, factor=4):
    """Validation set transforms

    Parameters
    ----------
    output_size: Resize the input image into square(width:output_size, height:output_size)
    factor: Scale down factor to apply on input image (default: 4)
    """
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        ScaleDown(factor),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])


def default_test_transforms():
    """Test set transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.410824894905, 0.370634973049, 0.359682112932],
            [0.278580576181, 0.26925137639, 0.27156367898]
        )
    ])


def display_transforms():
    """Transforms to conver the tensor back to PIL Image"""
    return transforms.Compose([
        DeNormalize(),
        transforms.ToPILImage()
    ])
