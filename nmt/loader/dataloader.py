import logging
import json
import h5py
import numpy as np
import torch
from nmt.utils import pload
import math

class textDataLoader(object):
    """
    Text data iterator class
    """
    def __init__(self, params, jobname):
        self.logger = logging.getLogger(jobname)
        self.src = params.get('src', None)
        if self.src == 'voice':
            self.ix_to_word = None
            self.vocab_size = 0
        else:
            infos = pload(params['infos'])
            self.ix_to_word = infos['itow']
            self.vocab_size = len(self.ix_to_word)
        self.ref = params["h5"]
        self.logger.info('Loading h5 file: %s' % params['h5'])
        self.logger.info('...Vocab size is %d ' % self.vocab_size)
        self.h5_file = h5py.File(params['h5'], 'r')
        raw = params.get('raw', False)
        self.label_dimensions = 2
        self.label_depth = 1
        print('Raw:', raw)
        if not raw:
            self.max_indices = {
                'train': len(self.h5_file["labels_train"]),
                'val': len(self.h5_file["labels_val"]),
                'test': len(self.h5_file["labels_test"])
                }
            self.label_dimensions = len(self.h5_file["labels_train"].shape)
            if self.label_dimensions >= 3:
                self.label_depth = self.h5_file["labels_train"].shape[2]
            self.logger.info('...Train:  %d | Dev: %d | Test: %d',
                             self.max_indices['train'],
                             self.max_indices['val'],
                             self.max_indices['test'])
            self.iterators = {'train': 0, 'val': 0, 'test': 0}
        else:
            print('Missing the usual splits')
            print('h5 keys:', list(self.h5_file))
            self.max_indices = {
                'full': len(self.h5_file["labels_full"]),
                }
            self.logger.info('...Full:  %d',
                             self.max_indices['full'])
            self.iterators = {'full': 0}

        self.batch_size = params['batch_size']
        # seq_length = params['max_length']
        # self.logger.warning('...Reading sequences up to %d', seq_length)
        self.max_src_length = params['max_src_length']
        self.max_trg_length = params['max_trg_length']
        if 'src_trg_ratio' in params:
            self.src_trg_ratio = float(params['src_trg_ratio'])
        else:
            self.src_trg_ratio = float(self.max_src_length) / float(self.max_trg_length)
        if not self.ix_to_word == None:
            word_to_ix = {w: ix for ix, w in self.ix_to_word.items()}
            self.pad = word_to_ix['<PAD>']
            self.unk = word_to_ix['<UNK>']
            try:
                self.eos = word_to_ix['<EOS>']
                self.bos = word_to_ix['<BOS>']
            except:
                self.eos = self.pad
                self.bos = self.pad
            try:
                self.blank = word_to_ix[' ']
            except:
                self.blank = 26
        else:
            self.pad = 0
            self.unk = 1
            self.eos = self.pad
            self.bos = self.pad

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    # def get_seq_length(self):
    #     return self.seq_length

    def get_src_batch(self, split, batch_size=None):
        seq_length = self.max_src_length
        batch_size = batch_size or self.batch_size
        label_batch = np.zeros([batch_size, seq_length, self.label_depth], dtype='float32')
        len_batch = []
        pointer = 'labels_%s' % split
        len_pointer = 'lengths_%s' % split
        max_index = self.max_indices[split]
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            label_batch[i] = self.h5_file[pointer][ri, :seq_length]
            len_batch.append(min(self.h5_file[len_pointer][ri], seq_length))

        #order = sorted(range(batch_size), key=lambda k: -len_batch[k])

        data = {}
        data['labels'] = torch.from_numpy(
            #label_batch[order, :max(len_batch)]
            label_batch[:, :max(len_batch)]
        ).cuda()

        data['lengths'] = torch.from_numpy(
            #np.array([len_batch[k] for k in order]).astype(int)
            np.array(len_batch).astype(int)
        ).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data, data['lengths']

    def get_trg_batch(self, split, src_len_batch, batch_size=None):
        seq_length = self.max_trg_length
        batch_size = batch_size or self.batch_size
        in_label_batch = np.zeros([batch_size, seq_length + 1], dtype='int')
        out_label_batch = np.zeros([batch_size, seq_length + 1], dtype='int')
        max_src_length = min(max(src_len_batch), self.max_src_length)
        if split == 'train':
            # max_src_length = self.h5_file['masks_train'].shape[1]
            alphabet_size = self.h5_file['masks_train'].shape[2]
            #masks_train_batch = np.zeros([batch_size, max_src_length, alphabet_size], dtype='int')
            masks_train_batch = np.ones([batch_size, max_src_length, alphabet_size], dtype='int')
        len_batch = []
        pointer = 'labels_%s' % split
        len_pointer = 'lengths_%s' % split
        max_index = self.max_indices[split]
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            # add <bos>
            in_label_batch[i, 0] = self.bos
            # in_label_batch[i, 1:] = self.h5_file[pointer][ri, :seq_length]
            src_len = min(src_len_batch[i], max_src_length)
            if split == 'train':
                masks_train_batch[i] = self.h5_file['masks_train'][ri, :src_len, :]
            full_str = self.h5_file[pointer][ri, :seq_length]
            no_blanks = full_str[(full_str != self.blank).nonzero()]
            no_blanks = no_blanks[(no_blanks > self.unk).nonzero()]
            ll = len(no_blanks)
            if ll >= src_len:
                ll = (src_len // self.src_trg_ratio).int().cpu()
            in_label_batch[i, 1:ll+1] = no_blanks[0:ll]
            len_batch.append(ll)
            out_label_batch[i] = np.insert(in_label_batch[i, 1:], ll, self.eos)

        data = {}
        data['labels'] = torch.from_numpy(in_label_batch[:, :max(len_batch) + 1]).cuda()
        data['out_labels'] = torch.from_numpy(out_label_batch[:, :max(len_batch)]).cuda()
        data['lengths'] = torch.from_numpy(
            np.array(len_batch).astype(int)
        ).cuda()
        data['src_lengths'] = src_len_batch
        if split == 'train':
            data['masks_train'] = torch.from_numpy(masks_train_batch).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

