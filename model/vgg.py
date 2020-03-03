import torch.nn as nn


__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]


config = {
    'A': [64, 'M', 128, 'M', \
          256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', \
          256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', \
          256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', \
          256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, config, num_classes, batch_norm=True):
        super(VGG, self).__init__()
        self.features = self.__make_layers(config, batch_norm)
        self.classifer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x

    def __make_layers(self, config, batch_norm):
        layers = list()
        in_channels = 3
        for v in config:
            if isinstance(v, int):
                conv = nn.Conv2d(in_channels, v, 3, 1, 1)
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
            else:
                layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


def vgg11(num_classes):
    return VGG(config['A'], num_classes)


def vgg13(num_classes):
    return VGG(config['B'], num_classes)


def vgg16(num_classes):
    return VGG(config['D'], num_classes)


def vgg19(num_classes):
    return VGG(config['E'], num_classes)
