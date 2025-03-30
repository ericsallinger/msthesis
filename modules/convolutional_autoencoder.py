import torch
import os
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO: fix lazy module init, swtich from upsample to deconvolution

class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        print(x)
        return x

class ConvAE(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int, channels: list, kernel: tuple = (3, 3)):
        super().__init__()

        self.log_stats = False
        
        # kernel size
        self.k = kernel

        # track shapes for upsampling
        self.shapes = []
        
        # encoder blocks
        w, h = input_shape
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(self._encoder_block(channels[i], channels[i+1]))
            w, h = self._wh_out(w), self._wh_out(h)

        self.flatten = nn.Flatten(1, -1)
        self.bn = nn.Sequential(
            nn.Linear(channels[-1]*w*h, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
            )
        
        # decoder blocks
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels[-1]*w*h),
            nn.BatchNorm1d(channels[-1]*w*h),
            nn.ReLU()
            )
        self.recon_shape = None

        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(self._decoder_block(channels[i], channels[i-1]))

    def _wh_out(self, whin, dilation=1, kernel_size=2, stride=2):
        whout = int((whin-dilation*(kernel_size-1)-1)/stride)+1
        return whout

    def _encoder_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=self.k, padding='same'),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, kernel_size=self.k, padding='same'),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _decoder_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=self.k, padding='same'),
            nn.BatchNorm2d(cin),
            nn.ReLU(),
            nn.Conv2d(cin, cout, kernel_size=self.k, padding='same'),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )

    def encode(self, x):
        self.shapes = []

        for encoder in self.encoders:
            self.shapes.append(x.shape[-2:]) 
            #print(f'Encoded tensor of shape: {self.shapes[-1]}')
            x = encoder(x)
            #print(f'----> {x.shape[-3:]}')
        self.recon_shape = x.shape
        #NOTE('ABOVE TENSOR SHAPE NEEDS TO BE RECONSTRUCTED')
        x = self.flatten(x)
        #print(f'Encoded tensor flattened to shape {x.shape}')

        x = self.bn(x)
        #print(f'Latent representation tensor shape: {x.shape}')
        return x

    def decode(self, x):
        x = self.fc(x)
        #print(f'Tensor shape output of fully connected layer {x.shape}')
        try:
            x = x.view(x.size(0), self.recon_shape[-3], self.recon_shape[-2], self.recon_shape[-1])
        except Exception as e:
            print(f'''{e}: .view() is hardcoded :( \n 
                  The latent representation needs to be reconstructed to the last tensor shape before the flatten operation. 
                  Check the first loop in the encode() function
                  ''')

        for decoder in self.decoders:
            #print(f'Decoded tensor of shape: {x.shape[-3:]}')
            x = decoder(x)
            #print(f'----> {self.shapes[-1]}')
            x = nn.Upsample(size=self.shapes.pop(), mode="bilinear", align_corners=True)(x)
        return x

    def forward(self, x):
        l_x = self.encode(x)
        r_x = self.decode(l_x)
        return r_x

if __name__ == "__main__":

    # locate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dimensionality of latent representation
    dim=32

    # each pair represents the number of input and output channels for one convolution block
    c = [1, 8, 16]

    # initialize model 
    conv_ae = ConvAE(input_shape=(30, 78), latent_dim=dim, channels=c)

    # move the model and its params to device
    conv_ae.to(device)

    # initialize optimizer with model parameters and learning rate
    optimizer = optim.Adam(conv_ae.parameters(), lr=1e-3)

    # model requires batch and channel dimension
    batch_size = 5
    num_channels = 1
    frame_length = 30
    input_dim = 78

    # move dataloader or input tensors to device
    input_tensor = torch.randn(batch_size, num_channels, frame_length, input_dim, device=device)

    print('input tensor shape ', input_tensor.shape)
    # forward pass through encoder and decoder for training
    output = conv_ae(input_tensor)

    # forward pass through only encoder for inference
    latent_re = conv_ae.encode(input_tensor)

    print(
        output.shape,
        latent_re.shape
    )

    # print(
    #     output
    # )
