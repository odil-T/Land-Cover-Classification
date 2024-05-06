"""
Implementation of the U-Net deep learning architecture in PyTorch.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = ConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.decoder1 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder2 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder3 = ConvBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder4 = ConvBlock(128, 64)

        # Final Layer
        self.final_conv = nn.Conv2d(64, output_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)

        x3 = self.encoder2(x2)
        x4 = self.pool2(x3)

        x5 = self.encoder3(x4)
        x6 = self.pool3(x5)

        x7 = self.encoder4(x6)
        x8 = self.pool4(x7)

        # Bottleneck
        bottleneck = self.bottleneck(x8)

        # Decoder
        x = self.up1(bottleneck)
        x = torch.cat([x, x7], dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.decoder2(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder4(x)

        # Final Layer
        x = self.final_conv(x)

        return x