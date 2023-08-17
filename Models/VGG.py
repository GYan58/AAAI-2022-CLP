import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

device = "cuda"

class VGGM(nn.Module):
    def __init__(self, features, size=512, out=10):
        super(VGGM, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers_m(cfg):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg_mnist():
    Model = VGGM(make_layers_m([64, 'M', 128, 'M', 256,256, 'M', 512,512, 'M', 512, 512,'M']))
    return Model



class VGG(nn.Module):
    def __init__(self, features, size=512, out=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG100(nn.Module):
    def __init__(self, features, size=512, out=100):
        super(VGG100, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg_cifar10():
    Model = VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']))
    #Model = VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']))
    return Model

def vgg_cifar100():
    Model = VGG100(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']))
    #Model = VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']))
    return Model
