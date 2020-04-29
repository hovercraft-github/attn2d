# -*- coding: utf-8 -*-
"""
Utilities
"""

import random
import pickle
import numpy as np
import torch


def pload(path):
    """
    Pickle load
    """
    return pickle.load(open(path, 'rb'),
                       encoding='iso-8859-1')


def pdump(obj, path):
    """
    Picke dump
    """
    pickle.dump(obj, open(path, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


def set_seed(seed):
    """
    Set seed for reproducibility
    """
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_contiguous(tensor):
    """
    Return a contiguous tensor
    Especially after: narrow() , view() , expand() or transpose()
    """
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def decode_sequence(ix_to_word, seq, eos, bos, remove_bpe=0):
    """
    Decode sequence into natural language
    Input: seq, N*T numpy array, with elements in 0 .. vocab_size.
    """
    N, T = seq.shape
    out = []
    for i in range(N):
        txt = []
        for j in range(T):
            ix = seq[i, j].item()
            if ix > 0 and not ix == eos:
                if ix == bos:
                    continue
                else:
                    txt.append(ix_to_word[ix])
            else:
                break
        sent = "".join(txt)
        if remove_bpe:
            sent = sent.replace('@@ ', '')
        out.append(sent)
    return out


def get_scores(logp, target):
    """
    Return scores per sentence
    """
    batch_size = logp.size(0)  # assume 1
    seq_length = logp.size(1)
    mask = target.gt(1).float()

    target = target[:, :seq_length]
    mask = mask[:, :seq_length]
    logp = to_contiguous(logp).view(-1, logp.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    ml_output = - logp.gather(1, target) * mask
    ml_output = torch.sum(ml_output) / torch.sum(mask)
    score = - ml_output
    # normalize
    # ml_output /= torch.sum(mask)
    return score


