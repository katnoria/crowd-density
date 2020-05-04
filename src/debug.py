import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from data import CustomEvalDataset, default_test_transforms

shha_train = f"/mnt/bigdrive/datasets/crowd-counting/C3Data/train_data/img/*.jpg"
shha_eval = f"/mnt/bigdrive/datasets/crowd-counting/C3Data/test_data/img/*.jpg"

# Collection of functions I find useful for debugging the model performance

def test_predict(model, loader, device):
    full_preds = []
    model.to(device)
    model.eval()
    for idx, (inputs, fname) in enumerate(loader):
        images = inputs.to(device)
        preds = model(images)
        preds = preds.data.cpu().numpy()
        count = np.sum(preds.reshape(preds.shape[0], -1) / 100., axis=1)
        full_preds.append({
            "fname": fname,
            "count": np.around(count)
        })

    return full_preds


def true_count(fname):
    den = np.genfromtxt(fname, delimiter=",").astype(np.float32)
    return int(np.sum(den))


def plot_sample(true_labels, pred_labels, max_val=10000):
    ys = []
    yhats = []
    for x, y in zip(true_labels, pred_labels):
        if (x <= max_val):
            ys.append(x)
            yhats.append(y)

    plt.figure(figsize=(6, 6))
    plt.scatter(ys, yhats, alpha=0.4)
    plt.plot([min(ys), max(ys)], [min(ys), max(ys)], 'k--', lw=4)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Model Evaluation: True vs Predicted")
    return plt

def load_predict(model, device, dirname, images=None, max_val=10000):
    """ Plots model performance

    Parameters
    ----------
    model: Instance of the trained model
    device: cuda or cpu
    dirname: Image folder to evaluate the model on
    images: List of images if you instead wish to evaluate on certain images
    max_val: Restrict the plot to a certain prediction size. This is useful
    when you are trying to assess the model performance for upto a certain 
    number of people (e.g see how model performs on images with < 200 people)
    """
    test_data = CustomEvalDataset(
        dirname, images, transform=default_test_transforms()
    )
    test_loader = DataLoader(
        test_data, batch_size=1, num_workers=10, shuffle=False
    )
    preds = test_predict(model, test_loader, device)
    true_labels = [
        true_count(
            record['fname'][0].replace("/img/", "/den/").replace(".jpg", ".csv")
        ) for record in preds
    ]
    # print(f"load_predict_plot: true_labels {true_labels}/")
    pred_labels = [record['count'][0] for record in preds]
    return (true_labels, pred_labels)


def load_predict_plot(model, device, dirname, images=None, max_val=10000):
    """ Plots model performance

    Parameters
    ----------
    model: Instance of the trained model
    device: cuda or cpu
    dirname: Image folder to evaluate the model on
    images: List of images if you instead wish to evaluate on certain images
    max_val: Restrict the plot to a certain prediction size. This is useful
    when you are trying to assess the model performance for upto a certain 
    number of people (e.g see how model performs on images with < 200 people)
    """
    (true_labels, pred_labels) = load_predict(model, device, dirname, images, max_val)
    return plot_sample(true_labels, pred_labels, max_val)
    

def display_loader(loader):
    data = next(iter(loader))
    img = data['image'].cpu().data[0].permute(1, 2, 0)
    img = img.numpy()
    den = data['den'].cpu().data[0]
    den = den.numpy()
    print(f"img: {img.shape}, den: {den.shape}, count: {int(np.sum(den))/100}")
    plt.imshow(img)
    plt.imshow(den[0], alpha=0.5)


def validate_dims(loader, test_model, input_size=224):
    data = next(iter(loader))
    img = data['image'].cpu().data[0]
    den = data['den'].cpu().data[0]
    output = test_model.forward(torch.rand(1, 3, input_size, input_size))
    print(f"img: {img.shape}, den: {den.shape}, output: {output.squeeze().shape}")
