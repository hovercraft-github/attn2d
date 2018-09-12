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
    Input: seq, N*D numpy array, with elements in 0 .. vocab_size.
    """
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = []
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0 and not ix == eos:
                if ix == bos:
                    continue
                else:
                    txt.append(ix_to_word[ix])
            else:
                break
        sent = " ".join(txt)
        if remove_bpe:
            sent = sent.replace('@@ ', '')
        out.append(sent)
    return out



