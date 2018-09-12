"""
DenseNet architecture
"""

from math import sqrt
import torch
import torch.nn as nn
from .dense_modules import _DenseLayer, _DenseLayer2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features,
                 kernel_size, bn_size,
                 growth_rate, drop_rate, gated,
                 bias, init_weights,
                 weight_norm,
                 layer_type=1):
        super(_DenseBlock, self).__init__()
        if layer_type == 1:
            LayerModule = _DenseLayer
        elif layer_type == 2:
            LayerModule = _DenseLayer2
        print('Dense channels:')
        for i in range(num_layers):
            print(">", num_input_features + i * growth_rate, end='')
            layer = LayerModule(
                num_input_features + i * growth_rate,
                growth_rate,
                kernel_size,
                bn_size,
                drop_rate,
                gated=gated,
                bias=bias,
                init_weights=init_weights,
                weight_norm=weight_norm,
                )
            self.add_module('denselayer%d' % (i + 1), layer)
        print()

    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()

    def track(self, x):
        activations = []
        for layer in list(self.children()):
            # layer is a DenseLayer
            x, newf = layer.track(x)
            activations.append(newf.data.cpu().numpy())
            x = torch.cat([x, newf], 1)
        return x, activations


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                bias=False))
        # std = sqrt(2/num_input_features)
        # self.conv.weight.data.normal_(0, std)
        # self.conv.bias.data.zero_()
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x, *args):
        return super(_Transition, self).forward(x)


class DenseNet(nn.Module):
    def __init__(self, num_init_features, params):
        super(DenseNet, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        block_config = params.get('num_layers', (6, 12, 24, 16))
        kernel_size = params.get('kernel', 3)
        bn_size = params.get('bn_size', 4)
        drop_rate = params.get('conv_dropout', 0)
        gated = params.get('gated', 0)
        bias = bool(params.get('bias', 1))
        init_weights = params.get('init_weights', 1)
        weight_norm = params.get('weight_norm', 0)
        half_init = params.get('half_inputs', 0)
        layer_type = params.get('layer_type', 1)

        self.features = nn.Sequential()
        num_features = num_init_features
        # start by halving the input channels
        if half_init:
            trans1 = nn.Conv2d(num_features, num_features // 2, 1)
            self.features.add_module('initial_transiton', trans1)
            num_features = num_features // 2

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                kernel_size=kernel_size,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                gated=gated,
                bias=bias,
                init_weights=init_weights,
                weight_norm=weight_norm,
                layer_type=layer_type
                )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans2 = _Transition(
                num_input_features=num_features,
                num_output_features=num_features // 2)

            self.features.add_module('transition%d' % (i + 1), trans2)
            num_features = num_features // 2

        self.output_channels = num_features
        # Final batch norm
        self.features.add_module('norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())

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

    def track(self, x):
        activations = []
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x, actv = layer.track(x)
                activations.append(actv)
            else:
                x = layer(x)
        return x, activations
