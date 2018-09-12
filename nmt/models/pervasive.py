"""
Pervasive attention
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import DenseNet
from .aggregator import Aggregator
from .embedding import Embedding
from .beam_search import Beam


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


class Pervasive(nn.Module):
    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
                 special_tokens):
        nn.Module.__init__(self)
        self.logger = logging.getLogger(jobname)
        self.version = 'conv'
        self.params = params
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = special_tokens['PAD']
        self.mask_version = params.get('mask_version', -1)
        # assert self.padding_idx == 0, "Padding token should be 0"
        self.bos_token = special_tokens['BOS']
        self.eos_token = special_tokens['EOS']
        self.kernel_size = params['network']['kernels'][
            0]  # assume using the same kernel size all over
        self.src_embedding = Embedding(
            params['encoder'],
            src_vocab_size,
            padding_idx=self.padding_idx
            )

        self.trg_embedding = Embedding(
            params['decoder'],
            trg_vocab_size,
            padding_idx=self.padding_idx,
            pad_left=True
            )

        self.input_channels = self.src_embedding.dimension + \
                              self.trg_embedding.dimension

        self.logger.info('Model input channels: %d', self.input_channels)
        self.logger.info("Selected network: %s", params['network']['type'])


        if params['network']['half_inputs']:
            self.logger.warning('Halfing the input channels')

        if params["network"]['type'] == "densenet":
            self.net = DenseNet(self.input_channels, params['network'])
            self.network_output_channels = self.net.output_channels
        else:
            raise ValueError(
                'Unknown architecture %s' % params['network']['type'])

        self.tie_target_weights = params['decoder']['tie_target_weights']
        self.copy_source_weights = params['decoder']['copy_source_weights']

        if self.tie_target_weights:
            self.logger.warning('Tying the decoder weights')
            last_dim = params['decoder']['input_dim']
        else:
            last_dim = None

        self.aggregator = Aggregator(self.network_output_channels, last_dim,
                                     params['aggregator'])
        self.final_output_channels = self.aggregator.output_channels

        self.prediction_dropout = nn.Dropout(
            params['decoder']['prediction_dropout'])
        self.logger.info('Output channels: %d', self.final_output_channels)
        self.prediction = nn.Linear(self.final_output_channels,
                                    self.trg_vocab_size)
        if self.copy_source_weights:
            self.trg_embedding.label_embedding.weight = self.src_embedding.label_embedding.weight
        if self.tie_target_weights:
            self.prediction.weight = self.trg_embedding.label_embedding.weight

    def init_weights(self):
        """
        Called after setup.buil_model to intialize the weights
        """
        if self.params['network']['init_weights']:
            nn.init.kaiming_normal(self.prediction.weight)

        self.src_embedding.init_weights()
        self.trg_embedding.init_weights()
        self.prediction.bias.data.fill_(0)
    
    # @profile
    def forward(self, data_src, data_trg):
        src_emb = self.src_embedding(data_src)
        trg_emb = self.trg_embedding(data_trg)
        Ts = src_emb.size(1)  # source sequence length
        Tt = trg_emb.size(1)  # target sequence length
        # 2d grid:
        src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        X = torch.cat((src_emb, trg_emb), dim=3)
        # del src_emb, trg_emb
        X = self._forward(X, data_src['lengths'])
        logits = F.log_softmax(
            self.prediction(self.prediction_dropout(X)), dim=2)
        return logits

    # @profile
    def _forward(self, X, src_lengths=None, mask=None):
        X = X.permute(0, 3, 1, 2)
        X = self.net(X)
        X = self.aggregator(X, src_lengths)
        return X

    def update(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net.update(X)
        attn = None
        if track:
            X, attn = self.aggregator(X, src_lengths, track=track)
        else:
            X = self.aggregator(X, src_lengths, track=track)
        return X, attn

    def track(self, data_src, kwargs={}):
        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)  # source sequence length
        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in range(batch_size)]).cuda()

        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # 2d grid:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        max_length = int(
            kwargs.get('max_length_a', 0) * Ts +
            kwargs.get('max_length_b', 50))

        seq = []
        alphas = []
        aligns = []
        activ_aligns = []
        activs = []
        embed_activs = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        E = torch.mm(self.prediction.weight.data,
                     self.aggregator.lin.weight.data)

        for t in range(max_length):
            X = torch.cat((src_emb, trg_emb), dim=3)
            Y, attn = self.update(X, data_src["lengths"], track=True)
            # align, activ_distrib, activ = attn
            if attn[0] is not None:
                alphas.append(attn[0])
            aligns.append(attn[1])
            activ_aligns.append(attn[2])
            activs.append(attn[3][0])
            cstr = torch.sum(E, dim=0).repeat(Ts, 1).cpu().numpy()
            clean_cstr = cstr[0]
            # cstr = cstr * attn[3][0]  # (N=1, Ts, d)
            embed_activs.append(None)
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remvoe pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1  # actually w/o +1, after that concat to have the desired width // train models with k=3 for less memory
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break

        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        # return seq, attention(max), activation & alphas(if attn)
        return seq, alphas, aligns, activ_aligns, activs, embed_activs, clean_cstr

    # @profile
    def sample(self, data_src, scorer, kwargs={}):
        beam_size = kwargs.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(data_src, kwargs)

        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)  # source sequence length
        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in range(batch_size)]).cuda()

        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # Coupling tensor:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        max_length = int(
            kwargs.get('max_length_a', 0) * Ts +
            kwargs.get('max_length_b', 50))

        seq = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in range(max_length):
            X = torch.cat((src_emb, trg_emb), dim=3)
            Y, _ = self.update(X, data_src["lengths"])
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remvoe pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1  # actually w/o +1, after that concat to have the desired width // train models with k=3 for less memory
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break

        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        return seq, None  # FIXME return scores as well

    def sample_beam(self, data_src, kwargs={}):
        # FIXME use update and reorder incremental states
        beam_size = kwargs['beam_size']
        src_labels = data_src['labels']
        src_lengths = data_src['lengths']
        batch_size = src_labels.size(0)
        beam = [Beam(beam_size, kwargs) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        Ts = src_labels.size(1)  # source sequence length
        src_labels = src_labels.repeat(beam_size, 1)
        src_lengths = src_lengths.repeat(beam_size, 1)
        # max_length = kwargs.get('max_length', 50)
        max_length = int(
            kwargs.get('max_length_a', 0) * Ts +
            kwargs.get('max_length_b', 50))
        for t in range(max_length):
            # Source:
            src_emb = self.src_embedding({
                'labels': src_labels,
                'lengths': None
            }).unsqueeze(1).repeat(1, t + 1, 1, 1)
            trg_labels_t = torch.stack([
                b.get_current_state() for b in beam if not b.done
            ]).t().contiguous().view(-1, 1)
            if t:
                # append to the previous tokens
                trg_labels = torch.cat((trg_labels, trg_labels_t), dim=1)
            else:
                trg_labels = trg_labels_t

            trg_emb = self.trg_embedding({
                'labels': trg_labels,
                'lengths': None
            }).unsqueeze(2).repeat(1, 1, Ts, 1)
            # X: N, Tt, Ts, Ds+Dt
            X = torch.cat((src_emb, trg_emb), dim=3)
            Y = self._forward(X, src_lengths)
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            word_lk = logits.view(beam_size, remaining_sents, -1).transpose(
                0, 1).contiguous()
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue
                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], t):
                    active += [b]
                # update trg_emb FIXME
                trg_labels_prev = trg_labels.view(beam_size, remaining_sents,
                                                  t + 1)
                trg_labels = trg_labels_prev[
                    beam[b].get_current_origin()].view(-1, t + 1)
            if not active:
                break
            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.contiguous().view(beam_size, remaining_sents,
                                                *t.size()[1:])
                new_size = list(view.size())
                new_size[1] = new_size[1] * len(active_idx) \
                    // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result.view(-1, result.size(-1))

            src_labels = update_active(src_labels)
            src_lengths = update_active(src_lengths)
            trg_labels = update_active(trg_labels)
            remaining_sents = len(active)

        # Wrap up
        allHyp, allScores = [], []
        n_best = 1
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            # hyps = list(zip(*[beam[b].get_hyp(k) for k in ks[:n_best]]))
            hyps = beam[b].get_hyp(ks[0])
            allHyp += [hyps]
        return allHyp, allScores



class Pervasive_Parallel(nn.DataParallel):
    """
    Wrapper for parallel training
    """
    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
                 special_tokens):
        model = Pervasive(jobname, params, src_vocab_size, trg_vocab_size,
                          special_tokens)
        nn.DataParallel.__init__(self, model)
        self.logger = logging.getLogger(jobname)
        self.version = 'conv'
        self.params = params
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.pad_token = special_tokens['PAD']
        # assert self.pad_token == 0, "Padding token should be 0"
        self.bos_token = special_tokens['BOS']
        self.eos_token = special_tokens['EOS']
        self.kernel_size = params['network']['kernels'][
            0]  # assume using the same kernel size all over

    def init_weights(self):
        self.module.init_weights()

    def _forward(self, X, src_lengths=None):
        return self.module._forward(self, X, src_lengths)

    def update(self, X, src_lengths=None):
        return self.module.update(X, src_lengths)

    def sample(self, data_src, scorer=None, kwargs={}):
        return self.module.sample(data_src, scorer, kwargs)
