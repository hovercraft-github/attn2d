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
        self.th_mask = 1  # both pad and unk
        self.normalize_batch = True
        self.eps = params['label_smoothing']
        self.version = 'label smoothed ml'

    def log(self):
        self.logger.info('Label smoothed ML loss with eps=%.2e' % self.eps)

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        mask = target.gt(self.th_mask).float()
        output, ml_loss = get_smooth_ml_loss(logp, target, mask,
                                             norm=self.normalize_batch,
                                             eps=self.eps)
        return {"final": output, "ml": ml_loss}, {}


class MLCriterion(nn.Module):
    """
    The default cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = 1  # both pad and unk
        self.normalize_batch = True
        self.penalize_confidence = params['penalize_confidence']
        self.version = 'ml'

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        mask = target.gt(self.th_mask).float()
        output = get_ml_loss(logp, target, mask,
                             norm=self.normalize_batch,
                             penalize=self.penalize_confidence)
        return {"final": output, "ml": output}, {}


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



