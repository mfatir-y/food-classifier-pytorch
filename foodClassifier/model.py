import torch.nn as nn


class FoodClassifierBase(nn.Module):
    """
    Base class for all food classifier models.
    Defines a shared interface so all models work with the same
    training loop, evaluation, and inference code.
    """
    def forward(self, x):
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BasicCNN(FoodClassifierBase):
    """
    A simple 4-block CNN trained from scratch.
    Good baseline for understanding what each layer contributes.

    Architecture:
        4x [Conv -> BatchNorm -> ReLU -> MaxPool]  (feature extractor)
        GlobalAvgPool -> Flatten -> Dropout -> Linear -> ReLU -> Linear  (classifier head)

    Channel progression: 3 -> 32 -> 64 -> 128 -> 256
    Spatial progression (128x128 input): 64 -> 32 -> 16 -> 8 -> 4 -> 1 (after avgpool)
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — detects low-level edges and colours
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # padding=1 keeps spatial size intact before pool
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # inplace=True saves memory
            nn.MaxPool2d(2),

            # Block 2 — detects textures and simple patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3 — detects shapes and object parts
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4 — detects high-level food features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Collapses any spatial size to 1x1 — makes model input-size agnostic
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):          # [B, 3, 128, 128]
        x = self.features(x)       # [B, 256, 8, 8]
        x = self.pool(x)           # [B, 256, 1, 1]
        x = x.flatten(start_dim=1) # [B, 256]
        x = self.classifier(x)     # [B, num_classes]
        return x


class DeepCNN(FoodClassifierBase):
    """
    A deeper version of BasicCNN with 5 blocks and a wider classifier head.
    Try this after BasicCNN is working well as your next experiment.

    Adds:
        - A 5th conv block (256 -> 512 channels)
        - A wider classifier head (512 -> 1024 -> num_classes)
        - Higher dropout to compensate for more parameters
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # lighter dropout on second layer
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


# TODO: class ResNetCNN(FoodClassifierBase)
    

def get_model(name: str, num_classes: int = 10) -> FoodClassifierBase:
    """
    Factory function — lets train.py request a model by name
    without importing every class directly.

    Usage:
        model = get_model("basic")
        model = get_model("deep")
    """
    models = {
        "basic": BasicCNN,
        "deep":  DeepCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name](num_classes=num_classes)