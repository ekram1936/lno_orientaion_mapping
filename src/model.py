"""
Model definition for orientation prediction from diffraction patterns."""
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(
                out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(
                out_ch), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x): return self.block(x)


class OrientationCNN(nn.Module):
    """
    Input (B,1,256,256) -> Output (B,6) sin-cos for phi1, Phi, phi2.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(
                64, 128), ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 6),
        )

    def forward(self, x): return self.head(self.gap(self.features(x)))


def count_parameters(model): return sum(p.numel()
                                        for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, device):
    model = OrientationCNN(dropout=cfg["model"]["dropout"]).to(device)
    print(
        f"  OrientationCNN built — {count_parameters(model):,} trainable parameters")
    return model
