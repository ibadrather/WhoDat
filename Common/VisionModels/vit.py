import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_h_14,
    ViT_H_14_Weights,
)


def ViT_pt(input_channels: int = 3, num_classes: int = 1000) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    num_ftrs = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(num_ftrs, num_classes)

    return model


def ViT(input_channels: int = 3, num_classes: int = 1000) -> nn.Module:
    model = vit_b_16(weights=None)

    num_ftrs = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(num_ftrs, num_classes)

    return model
