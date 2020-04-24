"""
Setup the model and the loss criterion
"""
import nmt.loss as loss
from .seq2seq import Seq2Seq
from .pervasive import Pervasive, Pervasive_Parallel



def build_model(jobname, params, src_vocab_size, trg_vocab_size, trg_specials):
    ref = params['model']
    if ref == "seq2seq-attention":
        model = Seq2Seq(jobname, params,
                        src_vocab_size,
                        trg_vocab_size,
                        trg_specials)
    elif ref == "pervasive":
        model = Pervasive(jobname, params, src_vocab_size,
                          trg_vocab_size, trg_specials)
    elif ref == "pervasive-parallel":
        model = Pervasive_Parallel(jobname, params, src_vocab_size,
                                   trg_vocab_size, trg_specials)

    else:
        raise ValueError('Unknown model %s' % ref)

    model.init_weights()
    return model


def define_loss(jobname, params, trg_dict):
    """
    Define training criterion
    """
    ver = params['version'].lower()
    if ver == 'ctc':
        crit = loss.CTCCriterion(jobname, params)
    elif ver == 'ml':
        crit = loss.MLCriterion(jobname, params)
    elif ver == 'smooth_ml':
        crit = loss.SmoothMLCriterion(jobname, params)
    else:
        raise ValueError('unknown loss mode %s' % ver)
    crit.log()
    return crit


