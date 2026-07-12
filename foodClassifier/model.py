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


class ResNetTransfer(FoodClassifierBase):
    """
    ResNet-50 with pretrained ImageNet weights, fine-tuned for food classification.
    The backbone is frozen — only the classifier head is trained.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        import torchvision.models as models

        resnet = models.resnet50(weights="IMAGENET1K_V2")

        # Freeze all backbone layers — they already know edges, textures, shapes
        for param in resnet.parameters():
            param.requires_grad = False

        # Keep everything except the final classification layer
        # resnet.fc is the original Linear(2048 -> 1000) for ImageNet's 1000 classes
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs [B, 2048, 1, 1]

        # Replace with your own head for 10 food classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)       # [B, 2048, 1, 1]
        x = self.classifier(x)     # [B, num_classes]
        return x

    def unfreeze_backbone(self, layers: int = 1):
        """
        Unfreeze the last N layer groups of the backbone for fine-tuning.
        Call this after initial training converges, then retrain with a lower LR (1e-4).
        layers=1 unfreezes layer4, layers=2 unfreezes layer3+4, etc.
        """
        children = list(self.backbone.children())
        for child in children[-layers:]:
            for param in child.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze {layers} layer(s) — trainable parameters: {trainable:,}")    

def get_model(name: str, num_classes: int = 10) -> FoodClassifierBase:
    """
    Factory function — lets train.py request a model by name
    without importing every class directly.

    Usage:
        model = get_model("basic")
        model = get_model("deep")
        model = get_model("resnet")
    """
    models = {
        "basic": BasicCNN,
        "deep":  DeepCNN,
        "resnet": ResNetTransfer,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name](num_classes=num_classes)