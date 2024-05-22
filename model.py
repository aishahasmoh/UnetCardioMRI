# U-Net is an image segmentation model
# It classify's each pixel in the image into an 
# Our model must automatically determine all objects and their precise location and boundaries at a pixel level in the image.
# we have a binary classification problem where we have to classify each pixel into one of the two classes, Class 1: Salt or Class 2: Not Salt (or, in other words, sediment).

# architecture: encoder-decoder
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# 2 convolutions, one encoder layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # bias is false because batch normalization will cancel it anyway.
        self.conv = nn.Sequential( # same conv, relue, sameconv
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, conv_sizes=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        for size in conv_sizes:
            self.encoder.append(DoubleConv(in_channels, size))
            in_channels = size
        
        self.encode_to_decode = DoubleConv(conv_sizes[-1],conv_sizes[-1]*2)
        
        # decoder, include skip connections, 
        for size in reversed(conv_sizes):
            self.decoder.append(nn.ConvTranspose2d(size*2, size, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(size*2, size))

        self.final = nn.Conv2d(conv_sizes[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.encode_to_decode(x)
        skips.reverse()

        for i, layer in enumerate(self.decoder): # up and then double conv
            if (i % 2) == 0: # ConvTranspose2d layer
                x = layer(x)
                if x.shape != skips[i//2].shape:
                    x = TF.resize(x, size=skips[i//2].shape[2:])
                x = torch.cat((skips[i//2], x), dim=1) # concatenate the 
            else: # DoubleConv layer
                x = layer(x)
        
        return self.final(x)


def test_UNet_output_shape():
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(3, 1, 256, 256)  # Example input tensor
    output = model(x)
    assert output.shape == x.shape, "Output shape is incorrect"
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(3, 1, 51, 51)  # Example input tensor
    output = model(x)
    assert output.shape == x.shape, "Output shape is incorrect"

def test_UNet_layers():
    model = UNet()
    layers = dict(model.named_children())
    expected_layers = ['encoder', 'decoder', 'pool', 'encode_to_decode', 'final']
    assert all(layer in layers for layer in expected_layers), "Missing layers in the model"

def test_UNet_encoder_decoder_sizes():
    model = UNet()
    encoder_sizes = [64, 128, 256, 512]
    decoder_sizes = encoder_sizes[::-1]
    assert len(model.encoder) == len(encoder_sizes), "Encoder size mismatch"
    assert len(model.decoder) // 2 == len(decoder_sizes), "Decoder size mismatch"
    for i, layer in enumerate(model.encoder):
        assert layer.conv[0].out_channels == encoder_sizes[i], f"Encoder layer {i} output size mismatch"
    for i, layer in enumerate(model.decoder):
        if i % 2 == 0:
            assert layer.out_channels == decoder_sizes[i//2], f"Decoder ConvTranspose2d layer {i//2} output size mismatch"
