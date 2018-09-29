"""
Variants of dense layers
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d import MaskedConv2d, GatedConv2d, AsymmetricMaskedConv2d


def _setup_conv_dilated(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    dilation = params.get('dilation', 2)
    print('Dilation: ', dilation)

    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(
        num_input_features,
        interm_features,
        kernel_size=1,
        bias=bias)
    conv2 = CV(
        interm_features,
        interm_features,
        kernel_size=kernel_size,
        bias=bias)

    conv3 = CV(
        interm_features,
        growth_rate,
        kernel_size=kernel_size,
        bias=bias,
        dilation=dilation)

    if init_weights == "manual":
        if not first:
            # proceeded by dropout and relu
            cst = 2 * (1 - drop_rate) 
        else:
            cst = 1
        # n_l = num_input_features 
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        # n_l = num_input_features * k * [(k-1)/2]
        # only relu
        std2 = sqrt(2 / (interm_featires * kernel_size *
                         (kernel_size - 1) // 2))
        conv2.weight.data.normal_(0, std2)
        conv3.weight.data.normal_(0, std2)

        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()
            conv3.bias.data.zero_()

    elif init_weights == "kaiming":
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv3.weight, mode="fan_out", nonlinearity='relu')

    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0) # dim = None ?
        conv2 = nn.utils.weight_norm(conv2, dim=0)
        conv3 = nn.utils.weight_norm(conv3, dim=0)

    return conv1, conv2, conv3


def _setup_conv(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    depthwise = params.get('depthwise', 0)

    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(
        num_input_features,
        interm_features,
        kernel_size=1,
        bias=bias)
    gp = growth_rate if depthwise else 1
    conv2 = CV(
        interm_features,
        growth_rate,
        kernel_size=kernel_size,
        bias=bias,
        groups=gp)

    if init_weights == "manual":
        # Init weights so that var(in) = var(out)
        if not first:
            # proceeded by dropout and relu
            cst = 2 * (1 - drop_rate) 
        else:
            cst = 1
        # n_l = num_input_features 
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        # n_l = num_input_features * k * [(k-1)/2]
        # only relu
        std2 = sqrt(2 / (bn_size * growth_rate * kernel_size *
                                   (kernel_size - 1) // 2))
        conv2.weight.data.normal_(0, std2)
        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()

    elif init_weights == "kaiming":
        #  Use pytorch's kaiming_normal_
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity='relu')

    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0) # dim = None ?
        conv2 = nn.utils.weight_norm(conv2, dim=0)

    return conv1, conv2


class _MainDenseLayer(nn.Module):
    """
    Main dense layer declined in 2 variants
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_size = params.get('bn_size', 4)
        self.growth_rate = params.get('growth_rate', 32)
        self.drop_rate = params.get('conv_dropout', 0.)
        
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
        # self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

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


class DenseLayer(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > BN > ReLU > Conv(k)
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2
            )


class DenseLayer_midDP(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > Dropout > BN > ReLU > Conv(k)
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.Dropout(p=self.drop_rate, inplace=True),
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2
            )


class DenseLayer_noBN(_MainDenseLayer):
    """
    ReLU > Conv(1) > ReLU > Conv(k)
    #TODO: check activ' var
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params, first=first)
        self.seq = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1,
            nn.ReLU(inplace=True),
            conv2
            )


class DenseLayer_Dil(_MainDenseLayer):
    """
    BN > ReLU > Conv(1)
    > BN > ReLU > Conv(k)
    > BN > ReLU > Conv(k, dilated)

    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2, conv3 = _setup_conv_dilated(num_input_features,
                                                  kernel_size,
                                                  params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv3
            )



class DenseLayer_Asym(nn.Module):
    """
    Dense layer with asymmetric convolution ie decompose a 3x3 conv into
    a 3x1 1D conv followed by a 1x3 1D conv.
    As suggested in: 
    Efficient Dense Modules of Asymmetric Convolution for
    Real-Time Semantic Segmentation
    https://arxiv.org/abs/1809.06323
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.drop_rate = params.get('conv_dropout', 0.)
        bias = params.get('bias', 0)
        bn_size = params.get('bn_size', 4)
        growth_rate = params.get('growth_rate', 32)
        dim1 = bn_size * growth_rate
        dim2 = bn_size // 2 * growth_rate

        conv1 = nn.Conv2d(
            num_input_features,
            dim1,
            kernel_size=1,
            bias=False)

        pad = (kernel_size - 1) // 2
        conv2s = nn.Conv2d(
            dim1,
            dim2,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            bias=False)

        conv2t = AsymmetricMaskedConv2d(
            dim2,
            growth_rate,
            kernel_size=kernel_size,
            bias=False)
        
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            conv2s,
            conv2t
            )

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)
        # self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


