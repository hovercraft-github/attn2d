import torch.nn as nn
from .pooling import *


class Aggregator(nn.Module):
    def __init__(self, input_channls, force_output_channels=None, params={}):
        nn.Module.__init__(self)
        mode = params.get("mode", "max")
        mapping = params.get('mapping', 'linear')
        num_fc = params.get('num_fc', 1)
        self.output_channels = input_channls
        if mode == 'mean':
            self.project = average_code
        elif mode == 'max':
            self.project = max_code
        elif mode == 'truncated-max':
            self.project = truncated_max
        elif mode == 'truncated-mean':
            self.project = truncated_mean
        elif mode == "max-attention":
            self.project = MaxAttention(params, input_channls)
            self.output_channels *= (2 - (params['first_aggregator'] == "skip"))
        else:
            raise ValueError('Unknown mode %s' % mode)
        self.add_lin = 0
        print('Aggregator:')
        if force_output_channels is not None:
            self.add_lin = 1
            # Map the final output to the requested dimension
            # for when tying the embeddings with the final projection layer
            assert self.output_channels > force_output_channels, "Avoid decompressing the channels"
            print(self.output_channels, end='')
            if num_fc == 1:
                lin = nn.Linear(self.output_channels, force_output_channels)
                print(">", force_output_channels)
            elif num_fc == 2:
                interm = (self.output_channels + force_output_channels ) // 2
                lin = nn.Sequential(
                        nn.Linear(self.output_channels, interm),
                        nn.ReLU(inplace=True),
                        nn.Linear(interm, force_output_channels)
                        )
                print(">", interm, ">", force_output_channels)
            else:
                raise ValueError('Not yet implemented')

            if mapping == "linear" :
                self.lin = lin 
            elif mapping == "tanh":
                self.lin = nn.Sequential(
                        lin,
                        nn.Tanh()
                        )
            elif mapping == "relu":
                self.lin = nn.Sequential(
                        lin,
                        nn.ReLU(inplace=True)
                        )
            self.output_channels = force_output_channels
            

    def forward(self, tensor, src_lengths, track=False, *args):
        if not track:
            proj = self.project(tensor, src_lengths, track, *args)
            proj = proj.permute(0, 2, 1)
            if self.add_lin:
                return self.lin(proj)
            else:
                return proj
        else:
            proj, attn = self.project(tensor, src_lengths, track, *args)
            proj = proj.permute(0, 2, 1)
            if self.add_lin:
                proj = self.lin(proj)
            return proj, attn




