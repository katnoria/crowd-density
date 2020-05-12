import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def plot_sample(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, 4, i + 1)

        image = sample['image']
        if type(image) == torch.Tensor:
            image = transforms.ToPILImage()(image)

        ax.imshow(image, alpha=0.9)
        den = sample['den']
        if len(den.shape) > 3:
            den = den[1:]
        ax.imshow(den, alpha=0.4)
        ax.axis('off')

        if i == 2:
            plt.show()
            break


def plot_image_den(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots(1, 2)

        image = sample['image']
        if type(image) == torch.Tensor:
            image = transforms.ToPILImage()(image)

        print(f"image: {image.size}")
        ax[0].imshow(image, alpha=0.9)
        den = sample['den']
        if len(den.shape) > 3:
            den = den[1:]
        print(f"den: {den.size}")
        ax[1].imshow(den, alpha=0.4)
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()
        break


def resize_mask(img, mask, use_img_size=False):
    (h, w) = img.shape[:-1] if len(img.shape) > 2 else img.shape
    (mw, mh) = mask.shape
    if (w <= h and w == mw) or (h <= w and h == mh):
        return img, mask

    if use_img_size:
        mh = h
        mw = w
    else:
        if w < h:
            mh = h
            mw = h
        else:
            mw = w
            mh = w
    return cv2.resize(img, (mw, mh), Image.BILINEAR), cv2.resize(mask, (mw, mh), Image.NEAREST)


def overlay_image_and_mask(img_fname, pred, count, figsize=(8, 8), use_img_size=False, alpha=0.6):
    img = np.asarray(Image.open(img_fname))
    new_img, new_mask = resize_mask(img, pred, use_img_size)
    if figsize is not None:        
        plt.figure(figsize=figsize)
    else:
        w, h = plt.figaspect(img)
        plt.figure(figsize=(w, h))

    plt.imshow(new_img)
    plt.imshow(new_mask, alpha=alpha)
    plt.axis('off')
    plt.title(f"People count: {count}")
    return plt


def error(true_labels, pred_labels):
    return np.array(true_labels) - np.array(pred_labels)


def mean_squared_error(true_labels, pred_labels):
    return np.around(np.mean(error(true_labels, pred_labels)**2))


def mean_absolute_error(true_labels, pred_labels):
    return np.around(np.mean(np.abs(error(true_labels, pred_labels))))
