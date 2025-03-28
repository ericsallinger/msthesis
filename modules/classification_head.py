import torch
import os
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.hidden_layers(x)
    
if __name__ == '__main__':
    batch_size = 5

    # batch of 5 random embedding vectors
    x = torch.randn((batch_size, 2048))
    
    # 5 random classes encoded by a one-hot vector
    y = F.one_hot(torch.randint(0, 3, (batch_size,)), num_classes=4).float()

    # define input dimension and number of classes to classify
    c = ClassificationHead(input_dim=2048, hidden_dim=512, num_classes=4)

    # vector containing probabilities
    y_hat = c(x)

    # compares predicted and ground truth distributions
    loss = F.cross_entropy(y_hat, y)

    print(loss, type(loss), loss.dtype)
