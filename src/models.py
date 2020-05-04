import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16Baseline(nn.Module):
    """Baseline model

    Baseline model that uses a pre-trained VGG16 network as backbone
    """

    def __init__(self, channels=[512, 128, 1], scale_factor=4):
        """
        Parameters
        ----------
        channels: Input channel size for all three layers
        scale_factor: Factor to upsample the feature map
        """
        super(VGG16Baseline, self).__init__()
        self.scale_factor = scale_factor
        conv_layers = list(models.vgg16(pretrained=True).features.children())
        # Mark the backbone as not trainable
        for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """ Forward pass

        Parameters
        ----------
        inputs: Batch of input images
        """
        output = self.model(inputs)
        output = F.upsample(output, scale_factor=self.scale_factor)
        return output


class VGG16WithDecoderV2(nn.Module):
    """ VGG16 Decoder"""

    def __init__(self):
        super(VGG16WithDecoderV2, self).__init__()
        conv_layers = list(models.vgg16(pretrained=True).features.children())[:23]
        # Pre-trained layers are not trainable
        for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=True, output_padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """forward pass
        
        Parameters
        ----------
        inputs: Batch of input images
        """
        output = self.model(inputs)
        return output
