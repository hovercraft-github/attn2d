# -*- coding: utf-8 -*-
"""
Regular  & label-smoothed cross-entropy losses
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nmt.utils import to_contiguous


class SmoothMLCriterion(nn.Module):
    """
    Label smoothed cross entropy loss
    """
    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = params.get('mask_threshold', 1)  # both pad and unk
        self.normalize_batch = bool(params.get('normalize_batch', 1))
        self.eps = params.get('label_smoothing', 0.1)
        self.version = 'label smoothed ml'

    def log(self):
        self.logger.info('Label smoothed ML loss with eps=%.2e' % self.eps)

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        mask = target.gt(self.th_mask).float()
        output, ml_loss = get_smooth_ml_loss(logp, target, mask,
                                             norm=self.normalize_batch,
                                             eps=self.eps)
        return {"final": output, "ml": ml_loss}, {}


class CTCCriterion(nn.Module):
    """
    CTC loss
    """
    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = params.get('mask_threshold', 1)  # both pad and unk
        self.version = 'ctc'
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False, reduction='none').cuda()
        # for param in self.parameters():
        #     param.requires_grad = False

    def log(self):
        self.logger.info('CTC loss')

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        batch_size = logp.size(0)
        seq_length = logp.size(1)
        #vocab = logp.size(2)
        labels = to_contiguous(target)
        labels = labels[:, :seq_length]
        y_lengths = (labels.size(1) - (labels <= self.th_mask).sum(dim=1)).int().cpu()
        labels = to_contiguous(labels).view(-1)
        labels = labels[(labels > self.th_mask).nonzero()].int()
        labels = labels.squeeze().cpu()

        x_lengths = torch.full((batch_size,), seq_length, dtype=torch.int32).cpu()  # Length of inputs
        logp = to_contiguous(logp).permute(1, 0, 2)

        output = self.ctc_loss(logp, labels, x_lengths, y_lengths)

        output = torch.sum(output)
        output /= batch_size

        if torch.isinf(output).any() or torch.isnan(output).any():
            output.data.copy_(torch.tensor(1000.0).data)
            # output_ = torch.zeros([1] * 0) + 1000
            # output_.requires_grad = True
            # output_.grad_fn = output.grad_fn
        return {"final": output, "ml": output}, {}

class MLCriterion(nn.Module):
    """
    The default cross entropy loss
    """
    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = params.get('mask_threshold', 1)  # both pad and unk
        self.normalize = params.get('normalize', 'ntokens')
        self.version = 'ml'

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        output = self.get_ml_loss(logp, target)
        return {"final": output, "ml": output}, {}

    def get_ml_loss(self, logp, target):
        """
        Compute the usual ML loss
        """
        # print('logp:', logp.size(), "target:", target.size())
        batch_size = logp.size(0)
        seq_length = logp.size(1)
        vocab = logp.size(2)
        target = target[:, :seq_length]
        logp = to_contiguous(logp).view(-1, logp.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = target.gt(self.th_mask)
        ml_output = - logp.gather(1, target)[mask]
        ml_output = torch.sum(ml_output)

        if self.normalize == 'ntokens':
            # print('initial ml:', ml_output.data.item())
            norm = torch.sum(mask)
            ml_output /= norm.float()
            # print('norm ml:', ml_output.data.item(), '// %d' % norm.data.item())
        elif self.normalize == 'seqlen':
            # print('initial ml:', ml_output.data.item())
            norm = seq_length
            ml_output /= norm
            # print('norm ml:', ml_output.data.item(), '// %d' % norm)
        elif self.normalize == 'batch':
            # print('initial ml:', ml_output.data.item())
            norm = batch_size
            ml_output /= norm
            # print('norm ml:', ml_output.data.item(), '// %d' % norm)

        else:
            raise ValueError('Unknown normalizing scheme')
        return ml_output



class MLCriterionNLL(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, job_name, params, pad_token):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.pad_token = pad_token
        self.normalize_batch = params['normalize_batch']
        self.penalize_confidence = params['penalize_confidence']
        self.sentence_avg = False
        self.version = 'ml'

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target, ntokens):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        logp = logp.view(-1, logp.size(-1))
        loss = F.nll_loss(logp, target.view(-1),
                          size_average=False,
                          ignore_index=self.pad_token,
                          reduce=True)

        print('loss pre norm:', loss.data.item())
        sample_size = target.size(0) \
                if self.sentence_avg else ntokens
        print('sample size:', sample_size)
        output = loss / sample_size
        print('returning:', output.data.item())
        return {"final": output, "ml": output}, {}

    def track(self, logp, target, mask, add_dirac=False):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        """
        # truncate to the same size
        N = logp.size(0)
        seq_length = logp.size(1)
        target = target[:, :seq_length].data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        target_d = np.zeros_like(logp)
        rows = np.arange(N).reshape(-1, 1).repeat(seq_length, axis=1)
        cols = np.arange(seq_length).reshape(1, -1).repeat(N, axis=0)
        target_d[rows, cols, target] = 1
        return logp, target_d

def get_ml_loss(logp, target, mask, scores=None,
                norm=True, penalize=0):
    """
    Compute the usual ML loss
    """
    # print('logp:', logp.size(), "target:", target.size())
    seq_length = logp.size(1)
    target = target[:, :seq_length]
    mask = mask[:, :seq_length]
    binary_mask = mask
    if scores is not None:
        # row_scores = scores.unsqueeze(1).repeat(1, seq_length)
        row_scores = scores.repeat(1, seq_length)
        mask = torch.mul(mask, row_scores)
    logp = to_contiguous(logp).view(-1, logp.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    if penalize:
        logp = logp.gather(1, target)
        neg_entropy = torch.sum(torch.exp(logp) * logp)
        ml_output = torch.sum(-logp * mask) + penalize * neg_entropy
    else:
        ml_output = - logp.gather(1, target) * mask
        ml_output = torch.sum(ml_output)

    if norm:
        ml_output /= torch.sum(binary_mask)
    return ml_output

def get_smooth_ml_loss(logp, target, mask,
                       norm=True, eps=0):
    """
    Cross entropy with label smoothing
    """
    # print('logp:', logp.size(), "target:", target.size())
    seq_length = logp.size(1)
    target = target[:, :seq_length]
    mask = mask[:, :seq_length]
    binary_mask = mask
    logp = to_contiguous(logp).view(-1, logp.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    ml_output = - logp.gather(1, target) * mask
    ml_output = torch.sum(ml_output)
    smooth_loss = -logp.sum(dim=1, keepdim=True) * mask
    smooth_loss = smooth_loss.sum() / logp.size(1)
    if norm:
        ml_output /= torch.sum(binary_mask)
        smooth_loss /= torch.sum(binary_mask)
    output = (1 - eps) * ml_output + eps * smooth_loss
    return output, ml_output
