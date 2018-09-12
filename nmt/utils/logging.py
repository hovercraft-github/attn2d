# -*- coding: utf-8 -*-

_ENDC = '\033[0m'
GREEN = '\x1b[32m'
YELLOW = '\x1b[33m'


def print_sampled(source, gt, pred, score=None):
    """
    Print translated sequences
    """
    source = " ".join(source.split()).encode('utf-8')
    gt = " ".join(gt.split()).encode('utf-8')
    pred = " ".join(pred.split()).encode('utf-8')
    print("SRC: ", source, GREEN, '\nTRG:', gt, YELLOW, "\nHYP: ", pred, _ENDC)
    return pred



