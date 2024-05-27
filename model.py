import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from config import *
# U-Net is an image segmentation model
# It classify's each pixel in the image into an 
# Our model must automatically determine all objects and their precise location and boundaries at a pixel level in the image.
# we have a binary classification problem where we have to classify each pixel into one of the two classes, Class 1: Salt or Class 2: Not Salt (or, in other words, sediment).

# architecture: encoder-decoder



# Encoder: multiple layers of (double convolution + max pooling)

# double convolution: 2 (same size convolutions + batch normalization + Relu) 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # bias in Conv2D is false because batch normalization cancels it anyway.
        self.conv = nn.Sequential( # same conv, normalize, relue, repeat
            nn.Conv2d(in_channels, out_channels, kernel_size=CONV_K, stride=1, padding=1, bias=False),
            nn.Dropout2d(p=DROPOUT),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=CONV_K, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, out_channels=NUM_CLASSES, conv_sizes=LAYER_SIZES):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=POOL_K, stride=POOL_S)

        # encoder: only save double convolution layers. max pooling has no parameters
        # layer1: (batch_size, width, height, in_channels) -> (batch_size, width, height, conv_sizes[0])
        # max_pooling1: (batch_size, width/2, height/2, conv_sizes[0]) 
        # layer12 (batch_size, width/2, height/2, conv_sizes[0]) -> (batch_size, width/2, height/2, conv_sizes[1]) 
        # max_pooling1: (batch_size, width/4, height/4, conv_sizes[1]) 

        for size in conv_sizes:
            self.encoder.append(DoubleConv(in_channels, size))
            in_channels = size

        self.encode_to_decode = DoubleConv(conv_sizes[-1],conv_sizes[-1]*2)

        # decoder: only save double convolution layers. downsampling between layers
        # layer1: (batch_size, width, height, in_channels) -> (batch_size, width, height, conv_sizes[0])
        # max_pooling1: (batch_size, width/2, height/2, conv_sizes[0]) 
        # layer12 (batch_size, width/2, height/2, conv_sizes[0]) -> (batch_size, width/2, height/2, conv_sizes[1]) 
        # max_pooling1: (batch_size, width/4, height/4, conv_sizes[1]) 

        # decoder, include skip connections,
        for size in reversed(conv_sizes):
            self.decoder.append(nn.ConvTranspose2d(size*2, size, kernel_size=UP_CONV_K, stride=UP_CONV_S))
            self.decoder.append(DoubleConv(size*2, size))

        # one final layer to return the segmentation maps.
        self.outputs = nn.Conv2d(conv_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = self.pool(x) # down sampling width and height by half

        x = self.encode_to_decode(x)
        skips.reverse()

        for i, layer in enumerate(self.decoder): # up and then double conv
            if (i % 2) == 0: # ConvTranspose2d layer
                x = layer(x)
                if x.shape != skips[i//2].shape: # crop to have matching sizes
                    x = TF.resize(x, size=skips[i//2].shape[2:])
                x = torch.cat((skips[i//2], x), dim=1) # concatenate 
            else: # DoubleConv layer
                x = layer(x)

        return self.outputs(x)

