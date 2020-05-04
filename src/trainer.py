import os
import copy
import time
import argparse
import logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from data import CCDataLoader, default_train_transforms, default_val_transforms
from models import VGG16Baseline, VGG16WithDecoderV2

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

today = datetime.today().strftime("%Y%m%d")
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger('cc-trainer')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(f"{LOG_DIR}/{today}.log")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

CKPT_DIR = "ckpt"
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)


class Trainer(object):
    """Trainer

    Trains the given model with given criterion and optimizer
    """

    def __init__(self, loaders, model, criterion, optimizer, name="vgg16", checkpoint=None):
        """
        Parameters
        ----------
        loaders: dict of train and val data loaders
        model: instance of model to train
        criterion: Choice of Loss (e.g nn.MSELoss)
        optimizer: Choice of optimizer (e.g SGD)
        name: Name of model (this will be used in model checkpoint filename)
        checkpoint: Location where to store checkpoints
        """
        self.loaders = loaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.name = name
        if checkpoint:
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict(ckpt['model_state_dict'])
        # Load on device
        self.model.to(device)
        logger.info(
            f"Trainer initialised with model: {self.model.__class__.__name__}, optimizer: {self.optimizer.__class__.__name__}")

    def save_model(self, loss, use_loss_in_name=False):
        """Saves the model

        Parameters
        ----------
        loss: model training loss
        use_loss_in_name: whether to use the loss info in checkpoint name    
        """
        fname = f"{self.name}-{loss:.4f}" if use_loss_in_name else f"{self.name}"
        fname = f"{fname}{today}.pt"
        torch.save({
            'epoch': 200,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, f"{CKPT_DIR}/{fname}")

    def train(self, epochs=20, writer_fname=None, tqdm_cls=None):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        writer = SummaryWriter(filename_suffix=self.name)
        if tqdm_cls:
            tqdm = tqdm_cls
        else:
            from tqdm import tqdm
        pbar = tqdm(range(epochs), "Training in progress")
        train_ds_size = len(self.loaders['train'])
        for idx, epoch in enumerate(pbar):
            logger.debug(f"Epoch {epoch}/{epochs}")

            self.model.train()

            running_loss = 0.0
            running_corrects = 0

            for imagedata in self.loaders['train'].dataloader:
                # put image and den to device
                image = imagedata['image'].to(device)
                den = imagedata['den'].to(device)                
                # zero all grads
                self.optimizer.zero_grad()

                # forward pass
                # track history in train phase
                with torch.set_grad_enabled(True):                    
                    outputs = self.model(image)   
                    if outputs.size()[-1] != den.size()[-1] or outputs.size()[-2] != den.size()[-2]:
                        import pdb; pdb.set_trace();

                    loss = None
                    if isinstance(self.criterion, list):
                        for criterion in self.criterion:
                            loss += criterion(outputs.squeeze(), den.squeeze())
                    else:
                        loss = self.criterion(outputs.squeeze(), den.squeeze())

                    loss.backward()
                    self.optimizer.step()

                # track stats
                # running_loss += loss.item()
                running_loss += loss.item() * image.size(0)
                # logger.debug(f"epoch: {epoch}, loss:{loss.item()}, running loss: {running_loss}")

            epoch_loss = running_loss / train_ds_size
            pbar.set_description(f"Training Loss:{epoch_loss:.4f}")

            if epoch % 10 == 0:
                for name, param in self.model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('loss/train', running_loss / train_ds_size, epoch)
            # validate
            epoch_val_loss = self.validate(writer, self.model, epoch)
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_val_loss)
    #         validate_on_fixed(model, epoch)
        elapsed = time.time() - since
        logger.info(f"Training complete in {elapsed//60:0f}m {elapsed%60:0f}s")
        logger.info(f"Best loss {best_loss}")
        # model.load_data_dict(best_model_wts)
        writer_fname = f"{self.model.__name__}.json" if writer_fname is None else writer_fname
        writer.export_scalars_to_json(writer_fname)
        writer.close()
        self.save_model(epoch_val_loss, use_loss_in_name=True)
        return self.model

    def validate(self, writer, model, epoch):
        model.eval()
        losses = []
        maes = []
        mses = []
        for imagedata in self.loaders['val'].dataloader:
            image = imagedata['image'].to(device)
            den = imagedata['den'].to(device)
            pred = model(image)
            # loss = self.criterion(pred.squeeze(), den.squeeze())
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    loss += criterion(pred.squeeze(), den.squeeze())
            else:
                loss = self.criterion(pred.squeeze(), den.squeeze())

            losses.append(loss.item())

            pred = pred.data.cpu().numpy()
            den = den.data.cpu().numpy()
            # get count
            pred_count = np.sum(pred) / 100
            den_count = np.sum(den) / 100
            # diff = den_count - pred_count
            diff = abs(pred_count - den_count)
            maes.append(diff)
            mses.append(diff**2)

        rmse = np.sqrt(np.mean(mses))
        logger.debug(f"{epoch}: Val loss:{np.mean(losses)}, mae:{np.mean(maes)}, mse: {rmse}")
        writer.add_scalar("val/loss", np.mean(losses), epoch)
        writer.add_scalar("val/mae", np.mean(maes), epoch)
        writer.add_scalar("val/mse", rmse, epoch)

        return np.mean(losses)


def main(args):
    # 1. Load data
    # Change this to the path where you store your training data
    data_dir = {
        "train": "/mnt/bigdrive/datasets/crowd-counting/C3Data/train_data/",
        "val": "/mnt/bigdrive/datasets/crowd-counting/C3Data/test_data/"
    }
    # data loader
    loaders = {
        "train": CCDataLoader(
            data_dir["train"],
            default_train_transforms(
                output_size=args.cropsize,
                factor=1
            ),
            num_workers=10
        ),
        "val": CCDataLoader(
            data_dir["val"],
            default_val_transforms(
                output_size=args.cropsize,
                factor=1
            ),
            num_workers=10
        )
    }

    # 2. Create model
    if args.model is None:
        # when transforms factor is 1, the following holds true
        # tip: use validate_dims from debug.py to find out the appropriate size
        model = VGG16Baseline(scale_factor=32)
    elif args.model == "vggbaseline":
        model = VGG16Baseline(scale_factor=32)
    elif args.model == "vggdecoder":
        model = VGG16WithDecoderV2()
    else:
        raise Exception(f"Unsupported model name {args.model} provided. Unable to continue further")

    # 3. Define loss
    criterion = nn.MSELoss()

    # 4. Define Optimizer and schedule (if required)
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Train model
    identifier = f"{args.model}-{lr}"
    logger.info(f"Trainer: Start with identifier {identifier}")
    trainer = Trainer(
        loaders,
        model,
        criterion,
        optimizer,
        name=identifier,
        checkpoint=args.checkpoint
    )
    model = trainer.train(args.epochs, writer_fname=identifier)
    logger.info("Trainer: Finished")


if __name__ == "__main__":
    # Prepare the logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # command-line args
    parser = argparse.ArgumentParser(description="Crowd Counter Trainer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path of model checkpoint to continue training an existing model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vggbaseline",
        help="Name of the model class"
    )
    parser.add_argument(
        "--cropsize",
        type=int,
        default=224,
        help="Crop input image to this size when training the model (default: 224)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate (default: 1e-5)"
    )
    # Not implemented
    # parser.add_argument(
    #     "--sched",
    #     type=float,
    #     default=1,
    #     help="learnig rate scheduler (default: 1)"
    # )    
    # Parse
    args = parser.parse_args()
    logger.debug(args)
    # Go!Go!Go
    main(args)
