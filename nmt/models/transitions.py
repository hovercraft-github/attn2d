from math import sqrt
import torch.nn as nn


class Transition(nn.Sequential):
    """
    Transiton btw dense blocks:
    BN > ReLU > Conv(k=1) to reduce the number of channels
    """
    def __init__(self, num_input_features, num_output_features, init_weights=0):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        conv = nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                bias=False)
        if init_weights == "manual":
            std = sqrt(2 / num_input_features)
            conv.weight.data.normal_(0, std)
        self.add_module('conv', conv)

    def forward(self, x, *args):
        return super(Transition, self).forward(x)


class Transition2(nn.Sequential):
    """
    Transiton btw dense blocks:
    ReLU > Conv(k=1) to reduce the number of channels

    """
    def __init__(self, num_input_features, num_output_features):
        super(Transition2, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                bias=False))
        
    def forward(self, x, *args):
        return super(Transition2, self).forward(x)

