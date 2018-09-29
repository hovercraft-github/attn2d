import torch as t
import torch.nn as nn


class MaskedConv1d(nn.Conv1d):
    """
    Masked (autoregressive) conv1d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, dilation=1,
                 groups=1, bias=False):
        # pad = (dilation * (kernel_size - 1)) // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           padding=padding,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        self.incremental_state = t.zeros(1, 1, 1)
        _, _, kH = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

    def update(self, x):
        k = self.kernel_size // 2 + 1
        buffer = self.incremental_state
        if buffer.size(-1) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1] = buffer[:, :, 1:].clone()
            buffer[:, :, -1:] = x[:, :, -1:]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output



