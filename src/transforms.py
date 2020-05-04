import random
import logging
import numpy as np
import cv2
from PIL import Image

from torchvision import transforms

# Custom Transforms to Handle Image an Density Map


class CenterCropOld(object):
    """ Centercrop image and density map

    Args:
        scale (tuple): Desired scale to apply before resizing
        output_size (int): Required crop size

    """

    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']

        # Original image size
        w, h = image.size
        left = int(round(w - self.output_size) / 2.)
        top = int(round(h - self.output_size) / 2.)
        right = int(round(w + self.output_size) / 2.)
        bottom = int(round(h + self.output_size) / 2.)
    
        # image = image.crop((x1, y1, x1 + self.output_size, y1 + self.output_size))
        # den = den[y1:y1 + self.output_size, x1:x1 + self.output_size]
        image = image.crop((left, top, right, bottom))
        den = den[top:bottom, left:right]
        return {"image": image, "den": den, 'fname': fname}

class CenterCrop(object):
    """ Centercrop image and density map

    Args:
        scale (tuple): Desired scale to apply before resizing
        output_size (int): Required crop size

    """

    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        output_w, output_h = self.output_size, self.output_size
        # Original image size
        # Crop Image
        w, h = image.size
        xi1 = int(round(w - self.output_size) / 2.)
        yi1 = int(round(h - self.output_size) / 2.)
        cropped_img = image.crop((xi1, yi1, xi1 + self.output_size, yi1 + self.output_size))

        # Crop density (argh the code is so messy)
        output_w = min(w, output_w)
        output_h = min(h, output_h)
        x1 = int(round(w - output_w) / 2.)
        y1 = int(round(h - output_h) / 2.)
        cropped_den = den[y1:y1 + output_h, x1:x1 + output_w]
        frame = np.full((self.output_size, self.output_size), 0.0)
        fx = int((self.output_size - cropped_den.shape[0])/2)
        fy = int((self.output_size - cropped_den.shape[1])/2)        
        frame[fx:fx+cropped_den.shape[0], fy:fy+cropped_den.shape[1]] = cropped_den[0:cropped_den.shape[0], 0:cropped_den.shape[1]]

        return {"image": cropped_img, "den": frame.astype('float32'), 'fname': fname}


class RandomFlip(object):
    """Flip image and density map"""
    def __init__(self, rnd=0.5):
        self.rnd = rnd

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']

        if random.random() < self.rnd:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            den = den[:, ::-1].copy()

        return {"image": image, "den": den, 'fname': fname}


class RandomGamma(object):
    """Gamma shift image and density map"""
    def __init__(self, gamma_range=(0.5, 1.5), rnd=0.3):
        self.gamma_range = gamma_range
        self.rnd = rnd

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']

        if random.random() < self.rnd:
            gamma = random.uniform(*self.gamma_range)
            image = image ** gamma

        return {"image": image, "den": den, 'fname': fname}


class ToTensor(object):
    """Convert Image and density map to tensor"""
    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']

        tfms = transforms.Compose([
            transforms.ToTensor()
        ])

        image = tfms(image)
        return {"image": image, "den": den, 'fname': fname}


class Normalize(object):
    """Normalize Image"""
    def __init__(self, mean=[0.410824894905, 0.370634973049, 0.359682112932],
                 std=[0.278580576181, 0.26925137639, 0.27156367898]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']

        tfms = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])

        image = tfms(image)
        den = np.reshape(den, [1, *den.shape])

        return {"image": image, "den": den, 'fname': fname}


class ScaleDown(object):
    """Scale-down the density map"""
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, sample):
        # C3 Paper: 2.2 Label Transformation (multiply by 64)
        image, den, fname = sample['image'], sample['den'], sample['fname']
        (w, h) = den.shape
        # Down-sample
        den = cv2.resize(den, (w // self.factor, h // self.factor), interpolation=cv2.INTER_CUBIC)
#         den = np.array(den.resize((w//self.factor, h//self.factor), Image.BICUBIC))
        # den = den * self.factor * self.factor
        return {"image": image, "den": den, 'fname': fname}


class LabelNormalize(object):
    """Normalize the density map
    C3 Paper suggests that network converges faster when we use
    a large number in density map (they suggest 100)
    """
    def __init__(self, norm=100):
        self.norm = norm

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        den = den * self.norm
        return {"image": image, "den": den, 'fname': fname}


class DeNormalize(object):
    """Denormalize image"""
    def __init__(self, mean=[0.410824894905, 0.370634973049, 0.359682112932],
                 std=[0.278580576181, 0.26925137639, 0.27156367898]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
