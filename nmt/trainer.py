# -*- coding: utf-8 -*-
"""
Trainer class
"""

import sys
import os.path as osp
import logging
import json
import time
import numpy as np

import torch
from tensorboardX import SummaryWriter
from nmt.utils import pload, pdump, set_seed
from nmt.models.evaluate import evaluate_model, evaluate_val_loss
from .optimizer import Optimizer,  LRScheduler
from ._trackers import TRACKERS


class Trainer(object):
    """
    Training a model with a given criterion
    """

    def __init__(self, jobname, params, model, criterion):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.params = params
        self.jobname = jobname

        self.logger = logging.getLogger(jobname)
        # reproducibility:
        set_seed(params['optim']['seed'])

        self.clip_norm = params['optim']['grad_clip']
        self.num_batches = params['optim']['num_batches']

        # Move to GPU
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # Initialize optimizer and LR scheduler
        self.optimizer = Optimizer(params['optim'], model)
        self.lr_patient = params['optim']['LR']['schedule'] == "early-stopping"
        if self.lr_patient:
            self.lr_patient = params['optim']['LR']['criterion']
            self.logger.info('updating the lr wrt %s', self.lr_patient)
        self.lr_scheduler = LRScheduler(params['optim']['LR'],
                                        self.optimizer.optimizer,
                                        )

        self.tb_writer = SummaryWriter(params['eventname'])
        self.log_every = params['track']['log_every']
        self.checkpoint = params['track']['checkpoint']
        self.evaluate = False
        self.done = False
        self.trackers = TRACKERS
        self.iteration = 0
        self.epoch = 0
        self.batch_offset = 0
        self.pass_no = 0
        # Dump  the model params:
        json.dump(params, open('%s/params.json' % params['modelname'], 'w'))

    def update_params(self, val_loss=None):
        """
        Update dynamic params:
        lr, scheduled_sampling probability and tok/seq's alpha
        """
        epoch = self.epoch
        iteration = self.iteration
        if not self.lr_patient:
            if self.lr_scheduler.mode in ["step-iter", "inverse-square",
                                          "cosine", 'shifted-cosine',
                                          'plateau-cosine']:
                self.lr_scheduler.step(iteration)
            else:
                self.lr_scheduler.step(epoch - 1)
        self.track('optim/lr', self.optimizer.get_lr())

    def step(self, data_src, data_trg, ntokens=0):
        """
        A signle forward step
        """
        # Clear the grads
        self.optimizer.zero_grad()
        batch_size = data_src['labels'].size(0)
        # evaluate the loss
        decoder_logit = self.model(data_src, data_trg)
        losses, stats = self.criterion(decoder_logit, data_trg['out_labels'])
        if not ntokens:
            ntokens = torch.sum(data_src['lengths'] *
                                data_trg['lengths']).data.item()

        # lmbda = torch.tensor(1e-2, device='cuda', requires_grad=False)
        # l2_reg = torch.tensor(0., device='cuda', requires_grad=False)
        # for param in self.model.parameters():
        #     l2_reg += torch.norm(param)
        # losses['final'] += (lmbda * l2_reg)
        # #losses['ml'] += lmbda * l2_reg

        return losses, batch_size, ntokens, decoder_logit

    def backward_step(self, loss, ml_loss, ntokens, nseqs, start, wrapped):
        """
        A single backward step
        """
        loss.backward()
        if self.clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.clip_norm)
            self.track('optim/grad_norm', grad_norm)
        self.track('optim/ntokens', ntokens)
        self.track('optim/batch_size', nseqs)

        self.optimizer.step()
        # torch.cuda.empty_cache()  # FIXME
        if np.isnan(loss.data.item()):
            sys.exit('Loss is nan')
        torch.cuda.synchronize()
        self.iteration += 1
        if wrapped:
            self.epoch += 1
        # Log
        if (self.iteration % self.log_every == 0):
            self.track('train/loss', loss.data.item())
            self.track('train/ml_loss', ml_loss.data.item())
            self.to_stderr(nseqs, ntokens, time.time()-start)
            self.tensorboard()

        self.evaluate = (self.iteration % self.checkpoint == 0)
        self.done = (self.epoch > self.params['optim']['max_epochs'])

    def validate(self, src_loader=None, trg_loader=None):
        """
        Evaluate on the dev set
        """
        params = self.params
        self.log('Evaluating the model on the validation set..')
        self.model.eval()
        if params.get('eval_bleu', 1):
            _, val_ml_loss, val_loss, bleu = evaluate_model(params['modelname'],
                                                            self,
                                                            src_loader,
                                                            trg_loader,
                                                            params['track'])
            self.log('BLEU: %.5f ' % bleu)
            self.track('val/perf/bleu', bleu)
            save_best = (self.trackers['val/perf/bleu'][-1] ==
                         max(self.trackers['val/perf/bleu']))
            save_every = 0

        else:
            val_ml_loss, val_loss = evaluate_val_loss(params['modelname'],
                                                      self,
                                                      src_loader,
                                                      trg_loader,
                                                      params['track'])
            save_every = 1
            save_best = 0

        self.track('val/loss', val_loss)
        self.track('val/ml_loss', val_ml_loss)
        self.tensorboard()
        # Save model if still improving on the dev set
        self.save_model(src_loader, trg_loader, save_best, save_every)
        self.model.train()
        if self.lr_patient == "loss":
            self.log('Updating the learning rate - LOSS')
            self.lr_scheduler.step(val_loss)
            self.track('optim/lr', self.optimizer.get_lr())
        elif self.lr_patient == "perf":
            assert not save_every
            self.log('Updating the learning rate - PERF')
            self.lr_scheduler.step(bleu)
            self.track('optim/lr', self.optimizer.get_lr())

    def save_model(self, src_loader, trg_loader, save_best, save_every):
        """
        checkoint model, optimizer and history
        """
        params = self.params
        modelname = params['modelname']
        checkpoint_path = osp.join(modelname, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        self.log("model saved to {}".format(checkpoint_path))
        optimizer_path = osp.join(modelname, 'optimizer.pth')
        torch.save(self.optimizer.state_dict(), optimizer_path)
        self.log("optimizer saved to {}".format(optimizer_path))
        self.trackers['src_iterators'] = src_loader.iterators
        self.trackers['trg_iterators'] = trg_loader.iterators
        self.trackers['iteration'] = self.iteration
        self.trackers['epoch'] = self.epoch
        pdump(self.trackers, osp.join(modelname, 'trackers.pkl'))

        if save_best:
            checkpoint_path = osp.join(modelname, 'model-best.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log("model saved to {}".format(checkpoint_path))
            optimizer_path = osp.join(modelname, 'optimizer-best.pth')
            torch.save(self.optimizer.state_dict(), optimizer_path)
            self.log("optimizer saved to {}".format(optimizer_path))
            pdump(self.trackers, osp.join(modelname, 'trackers-best.pkl'))

        if save_every:
            checkpoint_path = osp.join(modelname, 'model-%d.pth' % self.iteration)
            torch.save(self.model.state_dict(), checkpoint_path)
            self.log("model saved to {}".format(checkpoint_path))

    def load_checkpoint(self):
        """
        Load last saved params:
        for use with oar's idempotent jobs
        """
        params = self.params
        modelname = params['modelname']
        iterators_state = {}
        history = {}
        if osp.exists(osp.join(modelname, 'model.pth')):
            self.warn('Picking up where we left')
            # load model's weights
            saved_state = torch.load(osp.join(modelname, 'model.pth'))
            saved = list(saved_state)
            required_state = self.model.state_dict()
            required = list(required_state)
            del required_state
            if "module" in required[0] and "module" not in saved[0]:
                for k in saved:
                    kbis = "module.%s" % k
                    saved_state[kbis] = saved_state[k]
                    del saved_state[k]

            for k in saved:
                if "increment" in k:
                    del saved_state[k]
                if "transiton" in k:
                    kk = k.replace("transiton", "transition")
                    saved_state[kk] = saved_state[k]
                    del saved_state[k]
            self.model.load_state_dict(saved_state)
            # load the optimizer's last state:
            self.optimizer.load(
                torch.load(osp.join(modelname, 'optimizer.pth')
                           ))
            history = pload(osp.join(modelname, 'trackers.pkl'))
            iterators_state = {'src_iterators': history['src_iterators'],
                               'trg_iterators': history['trg_iterators']}

        elif params['start_from']:
            start_from = params['start_from']
            # Start from a pre-trained model:
            self.warn('Starting from %s' % start_from)
            if params['start_from_best']:
                flag = '-best'
                self.warn('Starting from the best saved model')
            else:
                flag = ''
            # load model's weights
            self.model.load_state_dict(
                    torch.load(osp.join(start_from, 'model%s.pth' % flag))
                    )
            # load the optimizer's last state:
            if not params['optim']['reset']:
                self.optimizer.load(
                    torch.load(osp.join(start_from, 'optimizer%s.pth' % flag)
                               ))
            history = pload(osp.join(start_from, 'trackers%s.pkl' % flag))
        self.trackers.update(history)
        self.epoch = self.trackers['epoch']
        self.iteration = self.trackers['iteration']
        return iterators_state

    def log(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def set_devices(self, devices):
        self.trackers['devices'].append(devices)
        self.trackers['time'].append(0)

    def increment_time(self, t):
        self.trackers['time'][-1] += t

    def track(self, k, v):
        """
        Track key metrics
        """
        if k not in self.trackers:
            raise ValueError('Tracking unknown entity %s' % k)
        if isinstance(self.trackers[k], list):
            self.trackers[k].append(v)
        else:
            self.trackers[k] = v
        self.trackers['update'].add(k)

    def tensorboard(self):
        """
        Write tensorboard events
        """
        for k in self.trackers['update']:
            self.tb_writer.add_scalar(k, self.trackers[k][-1], self.iteration)
        self.tb_writer.file_writer.flush()
        self.trackers['update'] = set()

    def to_stderr(self, batch_size, ntokens, timing):
        """
        Log to stderr
        """
        self.log('| epoch {:2d} '
                 '| iteration {:5d} '
                 '| lr {:02.2e} '
                 '| seq {:3d} '
                 '| sXt {:5d} '
                 '| ms/batch {:6.3f} '
                 '| total time {:6.2f} s'
                 '| loss {:6.3f} '
                 '| ml {:6.3f}'
                 .format(self.epoch,
                         self.iteration,
                         self.optimizer.get_lr(),
                         batch_size,
                         ntokens,
                         timing * 1000,
                         sum(self.trackers['time']),
                         self.trackers['train/loss'][-1],
                         self.trackers['train/ml_loss'][-1]))

