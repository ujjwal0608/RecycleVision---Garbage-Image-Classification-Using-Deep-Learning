
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.network = models.resnet50(pretrained=True)

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)
