import torch
import os
import pandas as pd
import json
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision.ops import SqueezeExcitation

import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# modules directory
from frame_dataloader_heavy import WorkloadFrame
import utils

class Conv(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int, channels: list, kernel: tuple):
        super().__init__()
        
        # kernel size and padding
        self.k = kernel

        # track shapes for upsampling
        self.shapes = []
        
        # encoder blocks
        h, w = input_shape
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(self._encoder_block(channels[i], channels[i+1]))
            h, w = self._wh_out(h), self._wh_out(w)

        self.flatten = nn.Flatten(1, -1)
        self.bn = nn.Sequential(
            nn.Linear(channels[-1]*w*h, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
            )
    
    def _wh_out(self, win, dilation=1, kernel_size=2, stride=2):
        whout = int(((win-dilation*(kernel_size-1)-1)/stride)+1)
        return whout
    
    def train(self, mode=True):
        super().train(mode)
        for m in self.encoders:
            m.train(mode)

    def eval(self):
        super().train(False)
        for m in self.encoders:
            m.train(False)


    def _encoder_block(self, cin, cout):
            sq_c = 2
            if cin != 1:
                return nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=self.k, padding='same'),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(),
                    nn.Conv2d(cout, cout, kernel_size=self.k, padding='same'),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(),
                    SqueezeExcitation(input_channels=cout, squeeze_channels=sq_c),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                return nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=self.k, padding='same'),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(),
                    nn.Conv2d(cout, cout, kernel_size=self.k, padding='same'),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))

    def encode(self, x):
        self.shapes = []
        for encoder in self.encoders:
            self.shapes.append(x.shape[-2:]) 
            #print(f'Encoded tensor of shape: {x.shape}')
            x = encoder(x)
            #print(f'----> {x.shape}')

        x = self.flatten(x)
        #print(f'Encoded tensor flattened to shape {x.shape}')

        x = self.bn(x)
        #print(f'Latent representation tensor shape: {x.shape}')
        return x
    

if __name__ == "__main__":

    # locate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dimensionality of latent representation
    ldim=32

    # dimensionality of hidden nn layer
    hdim = 16

    # each pair represents the number of input and output channels for one convolution block
    c = [1,8]

    # initialize model 
    conv_classifier = ConvClassifier(input_shape=(129, 5), latent_dim=ldim, channels=c, hidden_dim=hdim, kernel=(1, 4))

    # move the model and its params to device
    conv_classifier.to(device)

    # initialize optimizer with model parameters and learning rate
    optimizer = optim.Adam(conv_classifier.parameters(), lr=1e-3)

    #  file group: 'phys', 'cog', or 'tot'
    group='phys'

    # signal channel to resample to: 'temp', 'hrv, 'hr', 'hbo', 'eda'
    resample='temp'

    # size of sliding window relative to shortest signal length; always 50% overlap between windows
    context_length=0.5

    frames = WorkloadFrame(dir='files', group=group, resample=resample, context_length=context_length)
    frames_trainloader = DataLoader(frames, batch_size=64, num_workers=2,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)

    input = next(iter(frames_trainloader))[0].to(device)
    print('input tensor shape ', input.shape)

    # forward pass through only encoder for inference
    latent_re = conv_classifier.encode(input)

    print(
        conv_classifier.shapes,
        latent_re.shape
    )