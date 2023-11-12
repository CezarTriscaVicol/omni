import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, init_features=4, out_channels=1):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, "enc1")
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.encoder2 = UNet3D._block(features, 2*features, "enc2")
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.bottleneck = UNet3D._block(2*features, 4*features, "enc3")
        self.upconv2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2)

        self.decoder2 = UNet3D._block(4*features, 2*features, "dec2")
        self.upconv1 = nn.ConvTranspose3d(features*2, features  , kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(2*features, features, "dec1")
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
    
        #print("Input size: ", x.size())  # Print the size of the input tensor
        enc1 = self.encoder1(x)
        #print("After encoder1: ", enc1.size())  # Print the size after encoder1
        enc2 = self.encoder2(self.pool1(enc1))
        #print("After encoder2: ", enc2.size())  # Print the size after encoder2
        enc3 = self.bottleneck(self.pool2(enc2))
        #print("After bottleneck: ", enc3.size())  # Print the size after bottleneck
        dec2 = self.decoder2(torch.cat([enc2, self.upconv2(enc3)], 1))
        #print("After decoder2: ", dec2.size())  # Print the size after decoder2
        dec1 = self.decoder1(torch.cat([enc1, self.upconv1(dec2)], 1))
        #print("After decoder1: ", dec1.size())  # Print the size after decoder1
        pred = self.conv(dec1)
        #print("After final conv: ", pred.size())  # Print the size after final conv
        output = self.activation(pred)
        #print("After activation: ", output.size())  # Print the size after activation
        
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (name + "conv1", nn.Conv3d(in_channels, features, kernel_size=3, padding=1)),
                (name + "norm1", nn.BatchNorm3d(features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                # Add another convolution + batchnorm + relu sequence
                (name + "conv2", nn.Conv3d(features, features, kernel_size=3, padding=1)),
                (name + "norm2", nn.BatchNorm3d(features)),
                (name + "relu2", nn.ReLU(inplace=True))
            ]))