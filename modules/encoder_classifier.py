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

import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# modules directory
from frame_dataloader_heavy import WorkloadFrame
from convolutional_autoencoder import ConvAE
from classification_head import ClassificationHead
import utils

config = {'latent_dim': 32, 'conv_blocks': [1, 16], 'hidden_dim':16, 'kernel': (3, 2), 'id': 'TEST'}

class ConvClassifier(nn.Module):
    def __init__(self, latent_dim: int, channels: list, hidden_dim: int, kernel: tuple = (3, 3)):
        super().__init__()
        self.conv_e = ConvAE(latent_dim=latent_dim, channels=channels, kernel=kernel)
        self.