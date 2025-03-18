import torch
import os
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        print(x)
        return x

class ConvAE(nn.Module):
    def __init__(self, latent_dim: int, channels: list, kernel: tuple = (3, 3)):
        super().__init__()

        self.log_stats = False
        
        # only accept odd kernel sizes to preserve tensor shape with padding
        # if any(d % 2 == 0 for d in kernel):
        #     raise ValueError('Only odd kernel sizes accepted')
        
        # kernel size and padding
        self.k = kernel
        self.height_pad = kernel[0]//2
        self.width_pad = kernel[1]//2

        # track shapes for upsampling
        self.shapes = []
        
        # encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(self._encoder_block(channels[i], channels[i+1]))

        # latent space dynamically initialised on first forward pass
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten(1, -1)

        self.latent_shape = None
        self.recon_shape = None
        self.bn = None
        self.fc = None

        # decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(self._decoder_block(channels[i], channels[i-1]))

    def _encoder_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=self.k, padding=(self.height_pad, self.width_pad)),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, kernel_size=self.k, padding=(self.height_pad, self.width_pad)),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _decoder_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=self.k, padding=(self.height_pad, self.width_pad)),
            nn.BatchNorm2d(cin),
            nn.ReLU(),
            nn.Conv2d(cin, cout, kernel_size=self.k, padding=(self.height_pad, self.width_pad)),
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
        #print('ABOVE TENSOR SHAPE NEEDS TO BE RECONSTRUCTED')
        x = self.flatten(x)
        #print(f'Encoded tensor flattened to shape {x.shape}')

        if self.latent_shape is None:
            self.latent_shape = x.shape[-1]
            #print(f"Setting latent shape dynamically: {self.latent_shape}")
            self.bn = nn.Sequential(
                nn.Linear(self.latent_shape, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.ReLU()
                ).to(x.device)
            self.fc = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_shape),
                nn.LayerNorm(self.latent_shape),
                nn.ReLU()
                ).to(x.device)

        x = self.bn(x)
        #print(f'Latent representation tensor shape: {x.shape}')
        return x

    def decode(self, x):
        x = self.fc(x)
        #print(f'Tensor shape output of fully connected layer {x.shape}')
        try:
            x = x.view(x.size(0), self.decoders[0][0].in_channels, self.recon_shape[-2], self.recon_shape[-1])
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
    dim=2084

    # each pair represents the number of input and output channels for one convolution block
    c = [1, 64, 128, 256, 512]

    # initialize model 
    conv_ae = ConvAE(latent_dim=dim, channels=c)

    # move the model and its params to device
    conv_ae.to(device)

    # initialize optimizer with model parameters and learning rate
    optimizer = optim.Adam(conv_ae.parameters(), lr=1e-3)

    # model requires batch and channel dimension
    batch_size = 1
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

    print(
        output
    )
