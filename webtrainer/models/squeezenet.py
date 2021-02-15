"""
    Code for Squeezenet V1.1 from Pytorch docs.
    https://pytorch.org/docs/stable/_modules/torchvision/models/squeezenet.html#squeezenet1_1
    Paper: https://arxiv.org/pdf/1602.07360.pdf
    SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE
"""
import torch.nn as nn
import torch
import torch.nn.init as init
import enum
from collections import OrderedDict
from .resnet import BatchRenormalization2D
import torch.nn.functional as F


class ModelTasks(enum.Enum):
    classification = 1
    segmentation = 2


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, task=ModelTasks.classification, num_classes=1000, in_channels=3, out_channels=1):
        super(SqueezeNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.task = task

        if self.task.value == 1:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        elif self.task.value == 2:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # if self.task.value == 1:
        # Final convolution is initialized differently from the rest\
        if self.task.value == 1:
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif self.task.value == 2:
            self.init_segmentation_head(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.task.value == 1:
                    if m is final_conv:
                        init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform_(m.weight)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # Formward Propagation
    def forward(self, x):
        x = self.features(x)
        if self.task.value == 1:
            x = self.classifier(x)
            return torch.flatten(x, 1)
        elif self.task.value == 2:
            x = self.forward_seg_head(x)
            return x

    # Initializes segmentation head
    def init_segmentation_head(self, out_channels):
        self.upconv4_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.relu_s_4 = nn.PReLU()
        # For segmentation part we need batch normalization or else will be extremely difficult to train
        self.norm_4 = BatchRenormalization2D(num_features=256)

        self.upconv3_0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.relu_s_3 = nn.PReLU()
        # For segmentation part we need batch normalization or else will be extremely difficult to train
        self.norm_3 = BatchRenormalization2D(num_features=128)

        self.upconv2_0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.relu_s_2 = nn.PReLU()
        # For segmentation part we need batch normalization or else will be extremely difficult to train
        self.norm_2 = BatchRenormalization2D(num_features=64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        return

    # Does forward propagation of segmentation head
    def forward_seg_head(self, x):
        # Decoder block
        s_d4 = F.interpolate(x, scale_factor=2)
        s_d4 = self.upconv4_0(s_d4)
        s_d4 = self.relu_s_4(s_d4)
        s_d4 = self.upconv4_1(s_d4)
        s_d4 = self.norm_4(s_d4)
        # # Decoder block
        s_d3 = F.interpolate(s_d4, scale_factor=2)
        s_d3 = self.upconv3_0(s_d3)
        s_d3 = self.relu_s_3(s_d3)
        s_d3 = self.upconv3_1(s_d3)
        s_d3 = self.norm_3(s_d3)
        # # Decoder block
        s_d2 = F.interpolate(s_d3, scale_factor=2)
        s_d2 = self.upconv2_0(s_d2)
        s_d2 = self.relu_s_2(s_d2)
        s_d2 = self.upconv2_1(s_d2)
        s_d2 = self.norm_2(s_d2)
        conv = self.conv(s_d2)
        # Interpolate quickly
        conv = F.interpolate(conv, scale_factor=2)
        x = F.softmax(conv)
        return x
