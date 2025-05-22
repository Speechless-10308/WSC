'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

def normalized_thresh(z, mu=1.0):
    if len(z.shape) == 1:
        mask = (torch.norm(z, p=2, dim=0) < np.sqrt(mu)).float()
        return mask * z + (1 - mask) * F.normalize(z, dim=0) * np.sqrt(mu)
    else:
        mask = (torch.norm(z, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        return mask * z + (1 - mask) * F.normalize(z, dim=1) * np.sqrt(mu)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCon(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rotation=True, rotnet=None, classifier_bias=True,
                 loss_fn='con'):
        super(ResNetCon, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.output = nn.Linear(512*block.expansion, num_classes, bias=classifier_bias)
        self.channels = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=classifier_bias)
        channels = self.channels
        feat_dim = 2 * channels
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.BatchNorm1d(channels),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(channels, feat_dim))

        self.rotation = rotation
        if self.rotation:
            if rotnet is not None:
                self.rot = rotnet
            else:
                self.rot = nn.Linear(512 * block.expansion, 4)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        f = F.adaptive_avg_pool2d(out, 1)
        # c = self.output(f.squeeze())
        out = f.view(-1, self.channels)
        if self.training:
            feat_mlp = normalized_thresh(self.head(out))
            out = self.fc(out)
            return out, feat_mlp
        else:
            out = self.fc(out)
            return out


def ResNet18(**kwargs):
    return ResNetCon(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNetCon(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNetCon(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNetCon(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNetCon(Bottleneck, [3, 8, 36, 3], **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class resnet18con(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        dim = self.encoder.fc.in_features
        self.encoder.fc = Identity()

        self.fc = nn.Linear(dim, out_features=200)
        self.head =  nn.Sequential(nn.Linear(dim, 4 * dim), nn.BatchNorm1d(4 * dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(4 * dim, 128))

    def forward(self, x):
        out = self.encoder(x)
        feat = normalized_thresh(self.head(out))
        out = self.fc(out)
        if self.training:
            return out, feat
        else:
            return out



class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes=100):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features_in_dim = self.model.fc.in_features
        self.model.fc = Identity()
        self.fc = nn.Linear(self.features_in_dim, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self.features_in_dim, 4 * self.features_in_dim),
            nn.BatchNorm1d(4 * self.features_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.features_in_dim, 128)
        )
        
    def forward(self, x):
        out = self.model(x)
        feat = normalized_thresh(self.head(out))
        out = self.fc(out)
        if self.training:
            return out, feat
        return out