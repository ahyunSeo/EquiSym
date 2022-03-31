import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, inchannels=None):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if inchannels is not None:
            inplanes = inchannels
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, last_conv_only, last_convin = None, last_convout = None, use_bn=True, interpolate=0):
        super(Decoder, self).__init__()
        if backbone in ['resnet', 'resnet101', 'resnet50', 'resnet18']:
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.last_conv_only = last_conv_only
        if not last_conv_only:
            self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
            self.bn1 = BatchNorm(48)
            self.relu = nn.ReLU()
        
        self.interpolate = interpolate
        if use_bn:
            self.last_conv = nn.Sequential(nn.Conv2d(last_convin, last_convout, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(last_convout),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(last_convout, last_convout, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(last_convout),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(last_convout, num_classes, kernel_size=1, stride=1))
        else:
            self.last_conv = nn.Sequential(nn.Conv2d(last_convin, last_convout, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(last_convout),
                                        nn.ReLU(),
                                        nn.Conv2d(last_convout, last_convout, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(last_convout),
                                        nn.ReLU(),
                                        nn.Conv2d(last_convout, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, feat):
        if x is None:
            return self.last_conv(feat)

        if not self.last_conv_only:
            feat = self.conv1(feat)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            x = F.interpolate(x, size=feat.size()[2:], mode='bilinear', align_corners=True)

        if self.interpolate == 1:
            # baseline, (low_level_feat, feat), downsample (low_level_feat -> feat)
            x = F.interpolate(x, size=feat.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_aspp(backbone, output_stride, BatchNorm, inchannels=None):
    return ASPP(backbone, output_stride, BatchNorm, inchannels)

def build_decoder(num_classes, backbone, BatchNorm, last_conv_only, last_convin = None, last_convout = None, use_bn=True, interpolate=0):
    return Decoder(num_classes, backbone, BatchNorm, last_conv_only, last_convin, last_convout, use_bn, interpolate)

def build_backbone(backbone, output_stride, BatchNorm, pretrained=True):
    if backbone == 'resnet101':
        from modeling.resnet import ResNet101
        return ResNet101(output_stride, BatchNorm, pretrained)
    elif backbone == 'resnet18':
        from modeling.resnet import ResNet18
        return ResNet18(output_stride, BatchNorm, pretrained)
    elif backbone == 'resnet34':
        from modeling.resnet import ResNet34
        return ResNet34(output_stride, BatchNorm, pretrained)
    elif backbone == 'resnet50':
        from modeling.resnet import ResNet50
        return ResNet50(output_stride, BatchNorm, pretrained)
    else:
        raise NotImplementedError
