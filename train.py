#!/usr/bin/env python3

"""
Main training script
"""

import time
import logging
from nmt.params import parse_params, set_env
import nmt.utils.logging as lg
from nmt.utils import decode_sequence


def train(params):
    """
    Train NMT model
    """
    jobname = params['modelname']
    ngp = set_env(jobname, params['gpu_id'])
    import torch
    devices = {}
    for i in range(ngp):
        devices[i] = torch.cuda.get_device_name(i)
        print(devices[i])

    from nmt.loader import ReadData
    import nmt.models.setup as ms
    from nmt.trainer import Trainer

    logger = logging.getLogger(jobname)
    # Data loading:
    src_loader, trg_loader = ReadData({**params['data'], **params['encoder']}, params['modelname'])
    src_vocab_size = src_loader.get_vocab_size()
    trg_vocab_size = trg_loader.get_vocab_size()
    trg_specials = {'EOS': trg_loader.eos,
                    'BOS': trg_loader.bos,
                    'UNK': trg_loader.unk,
                    'PAD': trg_loader.pad,
                   }
    model = ms.build_model(jobname, params, src_vocab_size,
                           trg_vocab_size, trg_specials)
    logger.info('num. model params: %d', sum(p.data.numel()
                                             for p in model.parameters()))

    criterion = ms.define_loss(jobname, {**params['loss'], **params['data']}, trg_loader)
    trainer = Trainer(jobname, params, model, criterion)
    trainer.set_devices(devices)

    # Recover last checkpoint
    iters = trainer.load_checkpoint()
    src_loader.iterators = iters.get('src_iterators', src_loader.iterators)
    trg_loader.iterators = iters.get('trg_iterators', trg_loader.iterators)

    pass_no = 0

    # logger.warning("xy_ratio=%f", model.xy_ratio_ctc.sigmoid().squeeze()+1.0)

    if trainer.lr_patient:
        trainer.update_params()
    while True:
        # update parameters: lr, ...
        if not trainer.lr_patient:
            trainer.update_params()
        # if (trainer.iteration == 601) and (model.version == 'conv'):
        #     model.trg_embedding.label_embedding.weight.requires_grad_(False)
        torch.cuda.synchronize()
        avg_loss = torch.zeros(1).cuda()
        avg_ml_loss = torch.zeros(1).cuda()
        total_ntokens = 0
        total_nseqs = 0
        start = time.time()
        # Default num_batches=1
        for _ in range(params['optim']['num_batches']):
            data_src, order = src_loader.get_src_batch('train')
            data_trg = trg_loader.get_trg_batch('train', order)
            losses, batch_size, ntokens, logp = trainer.step(data_src, data_trg)
            avg_loss += ntokens * losses['final']
            avg_ml_loss += ntokens * losses['ml']
            total_nseqs += batch_size
            total_ntokens += ntokens
            pass_no += 1
            # batch_preds = torch.topk(logp, 1, dim=2)[1].squeeze()
            # n_pads = (batch_preds == 0).sum(dim=1)
            # print(n_pads)
            if pass_no % 1000 == 0:
                if model.version == 'fair':
                    batch_preds = model.sample(logp)
                    sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
                    sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
                    for (l, gl) in zip(sent_preds, sent_gold):
                        lg.print_sampled(gl, l)
                elif model.version == 'conv':
                    batch_preds = torch.topk(logp, 1, dim=2)[1].squeeze()
                    n_pads = (batch_preds == 0).sum(dim=1)
                    print(n_pads)
                    sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds,
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
                    sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    data_trg['out_labels'],
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)
                    for (l, gl) in zip(sent_preds, sent_gold):
                        lg.print_sampled(gl, l)
                    # logger.warning("xy_ratio=%f", model.xy_ratio_ctc.sigmoid().squeeze()+1.0)


        avg_loss /= total_ntokens
        avg_ml_loss /= total_ntokens

        trainer.backward_step(avg_loss, avg_ml_loss,
                              total_ntokens, total_nseqs,
                              start, data_src['bounds']['wrapped'])
        trainer.increment_time(time.time()-start)
        # Evaluate on validation set then save
        if trainer.evaluate:
            trainer.validate(src_loader, trg_loader)
        if trainer.done:
            logger.info('Max epochs reached!')
            break


if __name__ == "__main__":
    params = parse_params()
    train(params)
