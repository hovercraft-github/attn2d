from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .greedy_decoder import GreedyDecoder
from .embedding import NullEmbedding, Embedding

def _expand(tensor, dim, reps):
    # Expand 4D tensor in the source or the target dimension
    if dim == 1:
        return tensor.repeat(1, reps, 1, 1)
        # return tensor.expand(-1, reps, -1, -1)
    if dim == 2:
        return tensor.repeat(1, 1, reps, 1)
        # return tensor.expand(-1, -1, reps, -1)
    else:
        raise NotImplementedError

class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals

        TODO: use cuda if available

        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, jobname, params, num_classes):
        super(Wav2Letter, self).__init__()

        self.version = "fair"
        num_features = params['encoder']['input_dim']
        first_layer_stride = 2
        first_layer_kernel = 48
        last_layer_kernel = 32
        middle_layer_kernel = 7
        if params['model'] == "fair1_centroids":
            first_layer_stride = 1
            first_layer_kernel = 5
            middle_layer_kernel = 3
            last_layer_kernel = 7
        first_layer_padding = math.floor(first_layer_kernel/2)*2
        middle_layer_padding = math.floor(middle_layer_kernel/2)*2
        last_layer_padding = math.floor(last_layer_kernel/2)*2
        # Conv1d(in_channels, out_channels, kernel_size, stride)
        """         
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
        """        
        self.layers = nn.Sequential(
            #nn.Conv1d(num_features, 250, first_layer_kernel, first_layer_stride, padding=first_layer_padding),
            nn.Conv1d(num_features, 250, first_layer_kernel, first_layer_stride),
            torch.nn.ReLU(),
            #nn.BatchNorm1d(num_features=500),
            nn.Conv1d(250, 250, middle_layer_kernel),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel, padding=middle_layer_padding),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, middle_layer_kernel, padding=middle_layer_padding),
            torch.nn.ReLU(),
            #nn.BatchNorm1d(num_features=2000),
            #nn.BatchNorm1d(num_features=2000),
            #nn.Conv1d(250, 2000, last_layer_kernel, padding=last_layer_padding),
            nn.Conv1d(250, 2000, last_layer_kernel),
            torch.nn.ReLU(),
            #nn.BatchNorm1d(num_features=2000),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )
        self.padding_idx = 1
        self.src_embedding = NullEmbedding(
            params['encoder']
            )
        self.trg_embedding = Embedding(
            params['decoder'],
            num_classes,
            padding_idx=self.padding_idx
            )

        drpout = params['decoder']['prediction_dropout'] or .0
        self.prediction_dropout = None
        if drpout > 0:
            self.prediction_dropout = nn.Dropout(drpout)

    def init_weights(self):
        self.src_embedding.init_weights()

    def forward(self, data_src, data_trg):
        """Forward pass through Wav2Letter network that
            takes log probability of output

        Args:
            batch: mini batch of data
             shape (batch, frame_len, num_features)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, output_len, num_classes)
        """
        batch = self.src_embedding(data_src)
        batch = batch.permute(0,2,1)

        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)
        if not self.prediction_dropout == None:
            y_pred = self.prediction_dropout(y_pred)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)
        log_probs = log_probs.permute(0,2,1)

        return log_probs

    def sample(self, log_prob, scorer=None, kwargs={}):
        """
        Sample given source with keys:
            state - ctx - emb
        """        
        return GreedyDecoder(log_prob)

