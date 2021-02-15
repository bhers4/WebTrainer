import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
import torch.nn.functional as F
import enum

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ModelTasks(enum.Enum):
    classification = 1
    segmentation = 2


class ResNet(nn.Module):

    def __init__(self, block, layers, task, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, out_channels=1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        #
        self.task = task
        #
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        if self.task.value == 1:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.task.value == 2:
            self.init_segmentation_head(out_channels)
        else:
            print("Not classification or segmentation")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def init_segmentation_head(self, out_channels=1):
        self.upconv4_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder4 = ResNet._block(512, 256, "dec4", "")
        self.relu_s_4 = nn.PReLU()

        self.upconv3_0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder3 = ResNet._block(256, 128, "dec3", "")
        self.relu_s_3 = nn.PReLU()

        self.upconv2_0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder2 = ResNet._block(128, 64, "dec2", "")
        self.relu_s_2 = nn.PReLU()
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        return

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l1 = x
        x = self.layer2(x)
        l2 = x
        x = self.layer3(x)
        l3 = x
        x = self.layer4(x)
        if self.task.value == 1:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.task.value == 2:
            x = self.forward_seg_head(x, l3, l2, l1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def forward_seg_head(self, x, l3, l2, l1):
        # Decoder block
        s_d4 = F.interpolate(x, scale_factor=2)
        s_d4 = self.upconv4_0(s_d4)
        s_d4 = self.relu_s_4(s_d4)
        s_d4 = self.upconv4_1(s_d4)
        s_d4 = torch.cat((s_d4, l3), dim=1)  # This combines encoder4 and decoder 4
        s_d4 = self.decoder4(s_d4)
        # Decoder block
        s_d3 = F.interpolate(s_d4, scale_factor=2)
        s_d3 = self.upconv3_0(s_d3)
        s_d3 = self.relu_s_3(s_d3)
        s_d3 = self.upconv3_1(s_d3)
        s_d3 = torch.cat((s_d3, l2), dim=1)  # This combines encoder4 and decoder 4
        s_d3 = self.decoder3(s_d3)
        # Decoder block
        s_d2 = F.interpolate(s_d3, scale_factor=2)
        s_d2 = self.upconv2_0(s_d2)
        s_d2 = self.relu_s_2(s_d2)
        s_d2 = self.upconv2_1(s_d2)
        s_d2 = torch.cat((s_d2, l1), dim=1)  # This combines encoder4 and decoder 4
        s_d2 = self.decoder2(s_d2)
        conv = self.conv(s_d2)
        # Interpolate quickly
        conv = F.interpolate(conv, scale_factor=4)
        x = F.softmax(conv)
        return x

    def _block(in_channels, features, name, norm_name):
        if norm_name == "Group":
            modseq = nn.Sequential(OrderedDict([
                (name + "conv1",
                 nn.Conv2d(in_channels=in_channels,
                           out_channels=features,
                           kernel_size=3,
                           padding=1,
                           bias=True),),
                (name + "norm1", nn.GroupNorm(num_groups=16, num_channels=features)),
                (name + "relu1", nn.PReLU()),
                (name + "conv2",
                 nn.Conv2d(in_channels=features,
                           out_channels=features,
                           kernel_size=3,
                           padding=1,
                           bias=True),),
                (name + "norm2", nn.GroupNorm(num_groups=16, num_channels=features)),
                (name + "relu2", nn.PReLU()),
            ]))
            return modseq
        modseq = nn.Sequential(OrderedDict([
            (name + "conv1",
             nn.Conv2d(in_channels=in_channels,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=True),),
            (name + "norm1", BatchRenormalization2D(num_features=features)),
            (name + "relu1", nn.PReLU()),
            (name + "conv2",
             nn.Conv2d(in_channels=features,
                       out_channels=features,
                       kernel_size=3,
                       padding=1,
                       bias=True),),
            (name + "norm2", BatchRenormalization2D(num_features=features)),
            (name + "relu2", nn.PReLU()),
        ]))
        return modseq

class Resnet18(ResNet):
    def __init__(self, task=ModelTasks.classification , num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=3, out_channels=1):
        super(Resnet18, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2], task=task)
        block = BasicBlock
        layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        #
        self.task = task
        #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        if self.task.value == 1:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.task.value == 2:
            self.init_segmentation_head(out_channels)
        else:
            print("Not classification or segmentation")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def init_segmentation_head(self, out_channels):
        self.upconv4_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder4 = ResNet._block(512, 256, "dec4", "")
        self.relu_s_4 = nn.PReLU()

        self.upconv3_0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder3 = ResNet._block(256, 128, "dec3", "")
        self.relu_s_3 = nn.PReLU()

        self.upconv2_0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder2 = ResNet._block(128, 64, "dec2", "")
        self.relu_s_2 = nn.PReLU()
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        return

class Resnet34(ResNet):
    def __init__(self, task=ModelTasks.classification, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=3, out_channels=1):
        super(Resnet34, self).__init__(block=BasicBlock, layers=[3, 4, 6, 3], task=task)
        self.in_channels = in_channels
        block = BasicBlock
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # Save Task
        self.task = task

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if self.task.value == 1:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.task.value == 2:
            self.init_segmentation_head(out_channels)
        else:
            print("Not classification or segmentation")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def init_segmentation_head(self, out_channels):
        self.upconv4_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder4 = ResNet._block(512, 256, "dec4", "")
        self.relu_s_4 = nn.PReLU()

        self.upconv3_0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder3 = ResNet._block(256, 128, "dec3", "")
        self.relu_s_3 = nn.PReLU()

        self.upconv2_0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.upconv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True)
        self.decoder2 = ResNet._block(128, 64, "dec2", "")
        self.relu_s_2 = nn.PReLU()
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        return

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BatchRenormalization2D(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step=0.001):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum, requires_grad=False)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        # self.running_avg_mean = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=False)
        self.running_avg_mean = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=False)
        # self.running_avg_std = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=False)
        self.running_avg_std = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=False)
        self.init_flag = None

        self.max_r_max = 100.0
        self.max_d_max = 100.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor((1.0), requires_grad=False)
        self.d_max = torch.tensor((0.0), requires_grad=False)

    def forward(self, x):

        device = self.gamma.device

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True).to(device)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0, 2, 3), keepdim=True), self.eps, 1e10).to(device)

        if self.init_flag is None:
            self.running_avg_mean.data = batch_ch_mean
            self.running_avg_std.data = batch_ch_std
            self.init_flag = True

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.momentum = self.momentum.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)

        if self.training:
            r = torch.clamp(batch_ch_std / (self.running_avg_std), 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / (self.running_avg_std), -self.d_max, self.d_max).to(device).data.to(device)

            x = ((x - batch_ch_mean)*r)/batch_ch_std + d
            # x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

        else:
            x = (x - self.running_avg_mean.data) / self.running_avg_std.data

        x = self.gamma * x + self.beta

        if self.training:
            self.running_avg_mean.data = self.running_avg_mean.data + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
            self.running_avg_std.data = self.running_avg_std.data + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)
        return x