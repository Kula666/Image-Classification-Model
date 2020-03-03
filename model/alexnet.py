import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # nn.LocalResponseNorm(5),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # nn.LocalResponseNorm(5),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(num_classes):
    return AlexNet(num_classes=num_classes)
