"""
U-Net Architecture for E. coli Segmentation

Standard U-Net with encoder-decoder structure and skip connections.
Designed for 256x256 grayscale microscopy images.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2

    This is the basic building block of U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv

    Reduces spatial dimensions by 2x while increasing channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concatenate skip -> DoubleConv

    Increases spatial dimensions by 2x while decreasing channels.
    Uses bilinear interpolation for upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch due to odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to produce class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of E. coli growth stages.

    Architecture:
        Encoder: 4 down-blocks (64 -> 128 -> 256 -> 512)
        Bottleneck: 1024 channels
        Decoder: 4 up-blocks with skip connections (512 -> 256 -> 128 -> 64)
        Output: 1x1 conv to num_classes

    Input: (batch, 1, 256, 256) - grayscale microscopy images
    Output: (batch, num_classes, 256, 256) - per-pixel class logits

    Args:
        n_channels: Number of input channels (1 for grayscale)
        n_classes: Number of output classes (4: background, rod, dividing, microcolony)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 4, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 256 -> 256, channels: 1 -> 64
        x2 = self.down1(x1)   # 256 -> 128, channels: 64 -> 128
        x3 = self.down2(x2)   # 128 -> 64,  channels: 128 -> 256
        x4 = self.down3(x3)   # 64 -> 32,   channels: 256 -> 512
        x5 = self.down4(x4)   # 32 -> 16,   channels: 512 -> 512 (bilinear) or 1024

        # Decoder path with skip connections
        x = self.up1(x5, x4)  # 16 -> 32,   channels: 1024 -> 256
        x = self.up2(x, x3)   # 32 -> 64,   channels: 512 -> 128
        x = self.up3(x, x2)   # 64 -> 128,  channels: 256 -> 64
        x = self.up4(x, x1)   # 128 -> 256, channels: 128 -> 64

        # Output
        logits = self.outc(x) # 256 -> 256, channels: 64 -> n_classes

        return logits

    def predict(self, x):
        """Return predicted class indices for each pixel."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetSmall(nn.Module):
    """
    Smaller U-Net variant for faster training.

    Reduced channel counts: 32 -> 64 -> 128 -> 256 -> 512 (bottleneck)
    ~7.8M parameters vs ~31M for standard U-Net
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 4, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (half channels)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # Decoder
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)

        # Output
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def predict(self, x):
        """Return predicted class indices for each pixel."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing U-Net architectures...")

    # Test standard U-Net
    model = UNet(n_channels=1, n_classes=4)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"UNet: Input {x.shape} -> Output {y.shape}")
    print(f"UNet parameters: {model.count_parameters():,}")

    # Test small U-Net
    model_small = UNetSmall(n_channels=1, n_classes=4)
    y_small = model_small(x)
    print(f"UNetSmall: Input {x.shape} -> Output {y_small.shape}")
    print(f"UNetSmall parameters: {model_small.count_parameters():,}")
