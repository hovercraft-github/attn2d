import torch
import torch.nn as nn
import time


class AsymmetricMaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d kx1 kernel
    FIXME: particular case of the MaskedConv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        super().__init__(in_channels, out_channels,
                         (kernel_size, 1),
                         padding=(pad, 0),
                         groups=groups,
                         dilation=dilation,
                         bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super().forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class MaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           padding=pad,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class GatedConv2d(MaskedConv2d):
    """
    Gated version of the masked conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 bias=False, groups=1):
        super(GatedConv2d, self).__init__(in_channels,
                                          2*out_channels,
                                          kernel_size,
                                          dilation=dilation,
                                          bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super(GatedConv2d, self).forward(x)
        mask, out = x.chunk(2, dim=1)
        mask = self.sigmoid(mask)
        return out * mask



