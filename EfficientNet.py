# -*- coding:UTF-8 -*-
import copy
import math
from typing import Optional, Callable
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional


def _make_divisible(channel, divisor=8, min_channel=None):
    """
    make the channel equal its closest 8 times number.
    example: original channel is 8, the new channel is 2.2 times by original one, so it should be 16
    """
    if min_channel is None:
        min_channel = divisor
    new_channel = max(min_channel, int(channel + divisor / 2) // divisor * divisor)
    if new_channel < 0.9 * channel:
        new_channel += divisor
    return new_channel


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):  # inherit from Sequential needn't define forward function
    # This class will be used many times in MBconv.
    # It contains Convolution, BatchNormal and Activation.

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,  # DWConv or Conv
                 normal_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):

        padding = (kernel_size - 1) // 2
        if normal_layer is None:
            normal_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               normal_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    # SE module
    def __init__(self,
                 input_channel: int,  # MBConv input channel
                 expand_channel: int,  # After expand the input channel
                 squeeze_factor: int = 4):  # squeeze input_channel
        super(SqueezeExcitation, self).__init__()
        squeeze_channel = input_channel // squeeze_factor  # Full connection channel
        # Full connection using Conv with 1 kernel_size
        self.fc1 = nn.Conv2d(expand_channel, squeeze_channel, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_channel, expand_channel, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class MBConvConfig:
    # MBConvolution configure
    @staticmethod
    def adjust_channels(channel: int, width_coefficient: float):
        return _make_divisible(channel=channel * width_coefficient)

    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_channel: int,
                 out_channel: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True
                 drop_rate: float,
                 index: str,  # 1a,2a,2b
                 width_coefficient: float):
        self.input_channel = self.adjust_channels(input_channel, width_coefficient)
        self.kernel = kernel
        self.expanded_channel = self.input_channel * expanded_ratio
        self.out_channel = self.adjust_channels(out_channel, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index


class MBConv(nn.Module):
    # MBConvolution

    def __init__(self,
                 cnf: MBConvConfig,
                 normal_layer: Callable[..., nn.Module]):  # BN layer
        super(MBConv, self).__init__()

        if cnf.stride not in [1, 2]:  # DWConv stride should be 1 or 2
            raise ValueError("illegal stride value.")

        # Shortcut branch, when the MBConv input shape same as output shape.
        # When the stride equals 1 means their HW are same.
        # When both channel of input and output equals means the channel is same.
        # So the input's and output's shape are same.
        self.use_shortcut_connect = (cnf.stride == 1 and cnf.input_channel == cnf.out_channel)

        layers = OrderedDict()
        activation_layer = nn.SiLU

        # expand 1x1 convolution
        if cnf.expanded_channel != cnf.input_channel:
            # expanded_ratio isn't equal 1, so both they shouldn't equal
            # otherwise it equals 1, it means that needn't expand.
            layers.update({"expand_conv": ConvBNActivation(cnf.input_channel,
                                                           cnf.expanded_channel,
                                                           kernel_size=1,
                                                           normal_layer=normal_layer,
                                                           activation_layer=activation_layer)})

        # depth_wise convolution
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_channel,
                                                  cnf.expanded_channel,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_channel,
                                                  normal_layer=normal_layer,
                                                  activation_layer=activation_layer)})
        # use SE module
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_channel,  # input the MBConv input channel
                                                   cnf.expanded_channel)})

            # project last 1x1 conv
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_channel,
                                                        cnf.out_channel,
                                                        kernel_size=1,
                                                        normal_layer=normal_layer,
                                                        activation_layer=nn.Identity)})  # placeholder
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_channel
        self.is_strided = cnf.stride > 1  # 1 False; 2 True

        # Use the Dropout layer only when using shortcut connections
        if self.use_shortcut_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        # Anyway use the dropout layer
        result = self.dropout(result)
        # add the input to the result after dropout
        if self.use_shortcut_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,  # classify
                 dropout_rate: float = 0.2,  # MBConv using shortcut dropout
                 drop_connect_rate: float = 0.2,  # stage 9 FC dropout
                 block: Optional[Callable[..., nn.Module]] = None,
                 normal_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # stage 2-8
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        # depth stage 2-8 repeats number
        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = MBConv

        if normal_layer is None:
            normal_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # width stage 2-8
        adjust_channels = partial(MBConvConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        bneck_conf = partial(MBConvConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        mb_conv_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            # the first repeat is default configure, the rest should change its stride and output channel
            for i in range(round_repeats(cnf.pop(-1))):
                # i=0 means the first repeat ,its stride should be default, the rest repeats' stride is 1
                if i > 0:
                    # the rest's strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                # the dropout ratio gradually grow from 0.0 to configure setting
                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                # generate the layer index
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                mb_conv_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     normal_layer=normal_layer)})

        # building inverted residual blocks
        for cnf in mb_conv_setting:
            layers.update({cnf.index: block(cnf, normal_layer)})

        # build top
        last_conv_input_channel = mb_conv_setting[-1].out_channel
        last_conv_output_channel = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_channel,
                                               out_planes=last_conv_output_channel,
                                               kernel_size=1,
                                               normal_layer=normal_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        # before the last full connection if dropout rate is not 0.
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_channel, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
