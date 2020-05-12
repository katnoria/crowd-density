import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from data import CCDataLoader, CustomEvalDataset, default_test_transforms, default_val_transforms
from utils import mean_absolute_error, mean_squared_error, error

shha_train = f"/mnt/bigdrive/datasets/crowd-counting/C3Data/train_data/img/*.jpg"
shha_eval = f"/mnt/bigdrive/datasets/crowd-counting/C3Data/test_data/img/*.jpg"

# Collection of functions I find useful for debugging the model performance

def test_predict_with_validation_loader(model, device, dirname, cropsize=224, factor=1):
    predicted_labels = []
    true_labels = []
    fnames = []

    model.to(device)
    model.eval()
    maes = []
    mses = []

    loader = CCDataLoader(
                dirname,
                default_val_transforms(
                    output_size=cropsize,
                    factor=factor
                ),
                num_workers=10,
                bs=1,
                shuffle=1
            )    
    for imagedata in loader.dataloader:
        image = imagedata['image'].to(device)
        den = imagedata['den'].to(device)
        fname = imagedata['fname']
        pred = model(image)

        pred = pred.data.cpu().numpy()
        den = den.data.cpu().numpy()
        # get count
        pred_count = np.sum(pred) / 100
        den_count = np.sum(den) / 100

        predicted_labels.append(pred_count)
        true_labels.append(den_count)
        fnames.append(fname)

        # diff = den_count - pred_count
        diff = abs(pred_count - den_count)
        maes.append(diff)
        mses.append(diff**2)

    rmse = np.sqrt(np.mean(mses))
    print(f"loss:{np.mean(mses)}, mae:{np.mean(maes)}, mse: {rmse}")    

    return fnames, true_labels, predicted_labels

def test_predict(model, loader, device):
    full_preds = []
    model.to(device)
    model.eval()
    for idx, (inputs, fname) in enumerate(loader):
        images = inputs.to(device)
        preds = model(images)
        preds = preds.squeeze().data.cpu().numpy()        
        # count = np.sum(preds.reshape(preds.shape[0], -1), axis=1)/100.
        count = np.sum(preds) / 100
        full_preds.append({
            "fname": fname,
            "count": np.around(count)
        })

    true_labels = [
        true_count(
            record['fname'][0].replace("/img/", "/den/").replace(".jpg", ".csv")
        ) for record in preds
    ]

    return full_preds, true_labels


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

def load_predict(model, device, dirname, transform=default_test_transforms(), images=None, max_val=10000):
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
        dirname, images, transform=transform
    )
    test_loader = DataLoader(
        test_data, batch_size=1, num_workers=10, shuffle=False
    )
    preds, true_labels = test_predict(model, test_loader, device)
    # print(f"load_predict_plot: true_labels {true_labels}/")
    # pred_labels = [record['count'][0] for record in preds]
    pred_labels = [record['count'] for record in preds]
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


def generate_plot_from_csv(
        name, dataset, ds_type, cropsize=224,
        dirname="predictions/debug-448/"):
    """ Generate plots from CSV

    Parameters
    ---------
    name: Model name (vgg16baseline or vgg16decoder)
    dataset: Dataset name (SHHA or SHHB)
    ds_type: Set type (train or test)
    cropsize: Input image crop size
    """
    fname = f"{dirname}/{name}_{dataset}_{ds_type}_predictions_{cropsize}.csv"
    df = pd.read_csv(fname)
    df['diff'] = df.true_labels - df.predicted_labels

    scatter = alt.Chart(df).mark_circle().encode(
        alt.X("true_labels"),
        alt.Y("predicted_labels"),
        alt.Tooltip(["true_labels", "predicted_labels"])
    )
    line = alt.Chart(df).mark_line().encode(
        alt.X('true_labels', title="True"),
        alt.Y('true_labels', title="Predicted"),
        color=alt.value('rgb(0,0,0)')
    )

    mse = mean_squared_error(
        df.true_labels.values,
        df.predicted_labels.values)

    mae = mean_absolute_error(
        df.true_labels.values,
        df.predicted_labels.values)

    chart = (scatter + line).properties(
        title=f"INPUT {cropsize}, {dataset}:{ds_type.upper()}, MSE: {mse} | MAE: {mae}"
    )
    return chart

