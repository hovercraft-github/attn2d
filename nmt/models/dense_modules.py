"""
Variants of dense layers
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d import MaskedConv2d, GatedConv2d


relu = lambda tensor : F.relu(tensor, inplace=True)


class _DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 growth_rate,
                 kernel_size=3,
                 bn_size=4,
                 drop_rate=0,
                 gated=False,
                 bias=False,
                 init_weights=True,
                 weight_norm=False,
                ):
        super().__init__()
        self.kernel_size = kernel_size
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

        if init_weights:
            std1 = sqrt(2 / num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(
                2 * (1 - drop_rate) / (bn_size * growth_rate * kernel_size *
                                       (kernel_size - 1) // 2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()

        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)

        # BN > ReLU > Conv(k=1) > BN > ReLU > Conv(k=3)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            conv2
            )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


class _DenseLayer2(nn.Module):
    def __init__(self,
                 num_input_features,
                 growth_rate,
                 kernel_size=3,
                 bn_size=4,
                 drop_rate=0,
                 gated=False,
                 bias=False,
                 init_weights=True,
                 weight_norm=False,
                ):
        super().__init__()
        self.kernel_size = kernel_size
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

        if init_weights:
            std1 = sqrt(2 / num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(
                2 * (1 - drop_rate) / (bn_size * growth_rate * kernel_size *
                                       (kernel_size - 1) // 2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()

        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)

        # ReLU > BN > Conv(k=1) > ReLU > BN > Conv(k=3)
        self.seq = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_input_features),
            conv1,
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(bn_size * growth_rate),
            conv2
            )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


