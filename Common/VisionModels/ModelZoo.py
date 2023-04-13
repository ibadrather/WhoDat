import torch.nn as nn
from .alexnet import AlexNet
from .googlenet import GoogLeNet
from .resnet import ResNet50, ResNet101, ResNet152
from .vgg import (
    vgg16,
    vgg19,
    vgg16_bn,
    vgg19_bn,
)
from .vit import ViT, ViT_pt


def get_model(arch, input_channels: int, num_classes: int) -> nn.Module:
    models = dict(
        alexnet=AlexNet(input_channels=input_channels, num_classes=num_classes),
        googlenet=GoogLeNet(input_channels=input_channels, num_classes=num_classes),
        resnet50=ResNet50(input_channels=input_channels, num_classes=num_classes),
        resnet101=ResNet101(input_channels=input_channels, num_classes=num_classes),
        resnet152=ResNet152(input_channels=input_channels, num_classes=num_classes),
        vgg16=vgg16(input_channels=input_channels, num_classes=num_classes),
        vgg19=vgg19(input_channels=input_channels, num_classes=num_classes),
        vgg16_bn=vgg16_bn(input_channels=input_channels, num_classes=num_classes),
        vgg19_bn=vgg19_bn(input_channels=input_channels, num_classes=num_classes),
        ViT=ViT(input_channels=input_channels, num_classes=num_classes),
        ViT_pt=ViT_pt(input_channels=input_channels, num_classes=num_classes),
    )

    if arch not in models.keys():
        raise ValueError("Invalid model architecture: {}".format(arch))

    return models[arch]


def models_available() -> list:
    """
    Get list of available models
    """
    return [
        "alexnet",
        "googlenet",
        "resnet50",
        "resnet101",
        "resnet152",
        "vgg16",
        "vgg19",
        "vgg16_bn",
        "vgg19_bn",
        # "vgg16_bn_pt",
    ]


def test():
    import os
    import torch

    data = torch.randn(1, 3, 224, 224)

    archs = [
        "alexnet",
        "googlenet",
        "resnet50",
        "resnet101",
        "resnet152",
        "vgg16",
        "vgg19",
        "vgg16_bn",
        "vgg19_bn",
        "vgg16_bn_pt",
    ]

    for arch in archs:
        os.system("clear")
        print(f"Testing {arch} model")

        model = get_model(arch, 3, 10)

        out = model(data)

        print(f"For {arch} the output shape is {out.shape}")


def test_vit():
    import torch

    data = torch.randn(10, 3, 224, 224)

    model = ViT(num_classes=9)
    out = model(data)
    print(f"For vit_b_16 the output shape is {out.shape}")


if __name__ == "__main__":
    # test()
    test_vit()
