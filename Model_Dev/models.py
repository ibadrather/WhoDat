import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load pre-trained ResNet50
        resnet50 = models.resnet50(pretrained=True)

        # Remove the last fully connected layer (classification head)
        modules = list(resnet50.children())[:-1]

        # Add the layers back to create a new model without the classification head
        self.resnet50 = nn.Sequential(*modules)

        # Create a new classification head for facial recognition
        self.classifier = nn.Linear(resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(50176, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
