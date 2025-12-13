"""
Custom CNN for patch classification.

Simple architecture designed for 64x64 grayscale patches.
4 classes: rod, dividing, microcolony, background.
"""

import torch
import torch.nn as nn


class PatchCNN(nn.Module):
    """
    Baseline CNN classifier for E. coli growth stage patches.

    Architecture:
    - 4 conv blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier

    Input: (batch, 1, 64, 64)
    Output: (batch, 4) logits
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8 -> 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=1)
