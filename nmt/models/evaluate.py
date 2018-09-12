# -*- coding: utf-8 -*-

"""Evaluation utils."""
import logging
from collections import Counter
import math
import time
import numpy as np
import torch
from nmt.utils import decode_sequence
import nmt.utils.logging as lg
from nmt.models.gnmt import GNMTGlobalScorer


def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a
    list of translation hypotheses and references.
    With the default settings, this computes the exact same
    score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n.
                  `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((order,))
    correct = np.zeros((order,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        hyp_length += len(hyp)
        ref_length += len(ref)

        for i in range(order):
            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))
            ref_ngrams = Counter(zip(*[ref[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram])
                              for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    def divide(x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.true_divide(x, y)
            z[~ np.isfinite(z)] = 0
        return z

    scores = divide(correct, total)

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length)) if hyp_length > 0 else 0.0
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)


def evaluate_val_loss(job_name, trainer, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    max_samples = eval_kwargs.get('max_samples', -1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    logger = logging.getLogger(job_name)

    # Switch to evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
    logger.warn('Evaluated %d samples in %.2f s', n, time.time()-start)
    return ml_loss_sum / loss_evals, loss_sum / loss_evals


def evaluate_model(job_name, trainer, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    max_samples = eval_kwargs.get('max_samples', -1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    logger = logging.getLogger(job_name)

    # Make sure to be in evaluation mode
    model = trainer.model
    crit = trainer.criterion
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    start = time.time()
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            if trainer.criterion.version == "seq":
                losses, stats = crit(model, source, data_trg)
            else:  # ML & Token-level
                # init and forward decoder combined
                decoder_logit = model.decoder(source, data_trg)
                losses, stats = crit(decoder_logit, data_trg['out_labels'])
            batch_preds, _ = model.sample(source, eval_kwargs)
        else:
            losses, stats = crit(model(data_src, data_trg), data_trg['out_labels'])
            batch_preds, _ = model.sample(data_src, eval_kwargs)

        loss_sum += losses['final'].data.item()
        ml_loss_sum += losses['ml'].data.item()
        loss_evals = loss_evals + 1
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'],
                                      eos=src_loader.eos,
                                      bos=src_loader.bos)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
    logger.warn('Evaluated %d samples in %.2f s', len(preds), time.time()-start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, ml_loss_sum / loss_evals, loss_sum / loss_evals, bleu_moses


def sample_model(job_name, model, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    remove_bpe = eval_kwargs.get('remove_bpe', True)
    logger = logging.getLogger(job_name)
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    start = time.time()
    lenpen_mode = eval_kwargs.get('lenpen_mode', 'wu')
    scorer = GNMTGlobalScorer(eval_kwargs['lenpen'], 0, 'none', lenpen_mode)

    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            batch_preds, _ = model.decoder.sample(source, scorer, eval_kwargs)
        else:
            batch_preds, _ = model.sample(data_src, scorer, eval_kwargs)

        torch.cuda.empty_cache()  # FIXME choose an optimal freq
        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos,
                                          remove_bpe=remove_bpe)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos,
                                         remove_bpe=remove_bpe)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'],
                                      eos=src_loader.eos,
                                      bos=src_loader.bos,
                                      remove_bpe=remove_bpe)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos,
                                    remove_bpe=remove_bpe)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        ix1 = data_src['bounds']['it_max']
        # ix1 = 20
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            break
        del sent_source, sent_preds, sent_gold, batch_preds
    logger.warn('Sampled %d sentences in %.2f s', len(preds), time.time() - start)
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, bleu_moses


def track_model(job_name, model, src_loader, trg_loader, eval_kwargs):
    """Evaluate model."""
    source = []
    preds = []
    ground_truths = []
    batched_alphas = []
    batched_aligns = []
    batched_activ_aligns = []
    batched_activs = []
    batched_embed_activs = []
    batch_size = eval_kwargs.get('batch_size', 1)
    assert batch_size == 1, "Batch size must be 1"
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    max_samples = eval_kwargs.get('max_samples', -1)
    eval_kwargs['BOS'] = trg_loader.bos
    eval_kwargs['EOS'] = trg_loader.eos
    eval_kwargs['PAD'] = trg_loader.pad
    eval_kwargs['UNK'] = trg_loader.unk
    print('src_loader ref:', src_loader.ref)
    remove_bpe = 'BPE' in src_loader.ref
    print('Removing bpe:', remove_bpe)
    logger = logging.getLogger(job_name)
    # Make sure to be in evaluation mode
    model.eval()
    offset = eval_kwargs.get('offset', 0)
    print('Starting from ', offset)
    src_loader.iterators[split] = offset
    trg_loader.iterators[split] = offset
    # src_loader.reset_iterator(split)
    # trg_loader.reset_iterator(split)
    n = 0
    while True:
        # get batch
        data_src, order = src_loader.get_src_batch(split, batch_size)
        data_trg = trg_loader.get_trg_batch(split, order, batch_size)
        n += batch_size
        if model.version == 'seq2seq':
            source = model.encoder(data_src)
            source = model.map(source)
            batch_preds, _ = model.decoder.sample(source, eval_kwargs)
        else:
            # track returns seq, alphas, aligns, activ_aligns, activs, embed_activs, clean_cstr
            batch_preds, alphas, aligns, activ_aligns, activs, embed_activs, C = model.track(data_src, eval_kwargs)
            batched_alphas.append(alphas)
            batched_aligns.append(aligns)
            batched_activ_aligns.append(activ_aligns)
            batched_activs.append(activs)
            batched_embed_activs.append(embed_activs)

        # Initialize target with <BOS> for every sentence Index = 2
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1),
                                          eos=trg_loader.eos,
                                          bos=trg_loader.bos,
                                          remove_bpe=False)[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                         eos=trg_loader.eos,
                                         bos=trg_loader.bos,
                                         remove_bpe=False)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      data_src['labels'].data.cpu().numpy(),
                                      eos=src_loader.eos,
                                      bos=src_loader.bos,
                                      remove_bpe=False)
        source.append(sent_source)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'].data.cpu().numpy(),
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos,
                                    remove_bpe=False)
        if not verbose:
            verb = not (n % 300)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        if max_samples == -1:
            ix1 = data_src['bounds']['it_max']
        else:
            ix1 = max_samples

        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    print('Sampled %d sentences' % len(preds))
    bleu_moses, _ = corpus_bleu(preds, ground_truths)

    return {'source': source,
            'preds': preds,
            'alpha': batched_alphas,
            'align': batched_aligns,
            'activ_align': batched_activ_aligns,
            'activ': batched_activs,
            'embed_activ': batched_embed_activs,
            'channels_cst': C,
            "bleu": bleu_moses,
            }
