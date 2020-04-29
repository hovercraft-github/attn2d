import logging
import json
import h5py
import numpy as np
import torch
from nmt.utils import pload

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
        self.h5_file = h5py.File(params['h5'])
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
        self.seq_length = params['max_length']
        self.logger.warning('...Reading sequences up to %d', self.seq_length)
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
        else:
            self.pad = 0
            self.unk = 1
            self.eos = self.pad
            self.bos = self.pad

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_src_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        if self.src == 'voice':
            label_batch = np.zeros([batch_size, self.seq_length, self.label_depth], dtype='float32')
        else:
            label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
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
            label_batch[i] = self.h5_file[pointer][ri, :self.seq_length]
            len_batch.append(min(self.h5_file[len_pointer][ri],
                                 self.seq_length))

        order = sorted(range(batch_size), key=lambda k: -len_batch[k])

        data = {}
        data['labels'] = torch.from_numpy(
            label_batch[order, :max(len_batch)]
        ).cuda()

        data['lengths'] = torch.from_numpy(
            np.array([len_batch[k] for k in order]).astype(int)
        ).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data, order

    def get_trg_batch(self, split, order, batch_size=None):
        batch_size = batch_size or self.batch_size
        # in_label_batch = np.zeros([batch_size, self.seq_length + 1], dtype='int')
        # out_label_batch = np.zeros([batch_size, self.seq_length + 1], dtype='int')
        in_label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
        out_label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
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
            # in_label_batch[i, 0] = self.bos
            # in_label_batch[i, 1:] = self.h5_file[pointer][ri, :self.seq_length]
            full_str = self.h5_file[pointer][ri, :self.seq_length]
            no_blanks = full_str[full_str > 2]
            ll = len(no_blanks)
            in_label_batch[i, 0:ll] = no_blanks
            # add <eos>
            # ll = min(self.seq_length, self.h5_file[len_pointer][ri])
            # len_batch.append(ll + 1)
            # out_label_batch[i] = np.insert(in_label_batch[i, 1:], ll, self.eos)
            len_batch.append(ll)
            out_label_batch[i] = in_label_batch[i]

        data = {}
        data['labels'] = torch.from_numpy(in_label_batch[order, :max(len_batch)]).cuda()
        data['out_labels'] = torch.from_numpy(out_label_batch[order, :max(len_batch)]).cuda()
        data['lengths'] = torch.from_numpy(
            np.array([len_batch[k] for k in order]).astype(int)
        ).cuda()

        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index, 'wrapped': wrapped}
        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

