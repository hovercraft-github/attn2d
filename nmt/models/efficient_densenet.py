# Adapted from:
# https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py

import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from .conv2d import MaskedConv2d, GatedConv2d
from .transitions import Transition


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 growth_rate,
                 kernel_size=3,
                 bn_size=4,
                 drop_rate=0,
                 gated=False,
                 bias=False,
                 init_weights=0,
                 weight_norm=False,
                 efficient=False):
        super(_DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        if gated:
            CV = GatedConv2d
        else:
            CV = MaskedConv2d
        conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            bias=bias)
        conv2 = CV(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=kernel_size,
            bias=bias)
        if init_weights == "manual":
            std1 = sqrt(2 / num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(
                2 * (1 - drop_rate) / (bn_size * growth_rate * kernel_size *
                                       (kernel_size - 1) // 2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()
        elif init_weights == "kaiming":
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity='relu')
            nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity='relu')
        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', conv1)
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', conv2)
        
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad
                                  for prev_feature in prev_features):
            # Does not compute intermediate values, but recompute them in the backward pass:
            # tradeoff btw memory & computation
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

    def reset_buffers(self):
        self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2.update(self.relu2(self.norm2(x)))
        return torch.cat([res, x], 1)

        
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features,
                 kernels, bn_size,
                 growth_rate, drop_rate, gated,
                 bias, init_weights,
                 weight_norm,
                 efficient=False):
        super(_DenseBlock, self).__init__()
        print('Dense channels:', num_input_features, end='')
        for i in range(num_layers):
            print(">", num_input_features + (i+1) * growth_rate, end='')
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                kernels[i],
                bn_size,
                drop_rate,
                gated=gated,
                bias=bias,
                init_weights=init_weights,
                weight_norm=weight_norm,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()



class Efficient_DenseNet(nn.Module):
    """ 
    efficient (bool):
    set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, num_init_features, params):
        super(Efficient_DenseNet, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        block_layers = params.get('num_layers', (6, 12, 24, 16))
        # kernel_size = params.get('kernel', 3)
        block_kernels = params['kernels']
        bn_size = params.get('bn_size', 4)
        drop_rate = params.get('conv_dropout', 0)
        gated = params.get('gated', 0)
        bias = bool(params.get('bias', 1))
        init_weights = params.get('init_weights', 0)
        weight_norm = params.get('weight_norm', 0)
        divide_channels = params.get('divide_channels', 2)
        efficient = params.get('efficient', 0)

        self.features = nn.Sequential()
        num_features = num_init_features
        # start by reducig the input channels
        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == "manual":
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels

        # Each denseblock
        for i, (num_layers, kernels) in enumerate(zip(block_layers,
                                                      block_kernels)):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                kernels=kernels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                gated=gated,
                bias=bias,
                init_weights=init_weights,
                weight_norm=weight_norm,
                efficient=efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = Transition(
                num_input_features=num_features,
                num_output_features=num_features // 2,
                init_weights=init_weights)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
            print("> (trans) ", num_features)

        self.output_channels = num_features
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return  self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                layer.reset_buffers()


