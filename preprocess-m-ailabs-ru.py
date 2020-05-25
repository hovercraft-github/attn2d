#!/usr/bin/env python3

"""
Main pre-processing script
"""

import os
import os.path as osp
import argparse
import h5py
import numpy as np
from nmt.utils import pdump
from pghelper import PgHelper
import torch


def build_vocab_from_db(max_words, vocab_file):
    """
    Build vocabulary
    """
    # Count occurrences of the certain dictionary item
    counts = {}
    # Count occurrences of sentences with the certain number of dictionary items
    sent_lengths = {}

    select = """
        select 	al.dic_item, al.n_times, al.ac_group
        from 	alphabet al
        order by al.ac_group, al.dic_item
        """
    with PgHelper() as pg_object:
        for row in pg_object.exec(select):
            counts[row[0]] = row[1]
    select = """
        select 	len n_chars, count(1) cnt_occurs
        from (
            select  length(trim(regexp_replace(lower(af.pronounced_text), '[\\''"]', '', 'g'))) len
            from 	audio_files af
                    join feature_counts fc on (fc.file_id = af.id and fc.cnt_bulk >= 40)
            ) le
        group by len
        """
    with PgHelper() as pg_object:
        for row in pg_object.exec(select):
            sent_lengths[row[0]] = row[1]

    # print some stats
    total_words = sum(counts.values())
    print('total letters:', total_words)
    vocab = [w for w, _ in counts.items()]

    print('number of letters in alphabet would be %d' % (len(vocab), ))
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)

    print('inserting the special UNK token')
    #vocab.insert(0, "<BOS>")
    #vocab.insert(0, "<EOS>")
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<PAD>")
    # writing a vocab file:
    with open(vocab_file, 'w') as fv:
        for word in vocab:
            fv.write(word+'\n')
    # Dump the statistics for later use:
    pdump({"counts": counts,
           "vocab": vocab,
           "bad words": 0,
           "lengths": sent_lengths},
          vocab_file + ".stats")

    return vocab

def encode_sentences(sentences, params, wtoi):
    """
    encode all sentences into one large array, which will be 1-indexed.
    No special tokens are added, except from the <pad> after the effective length
    """
    max_length_trg = params.max_length_trg
    lengths = []
    m = len(sentences)
    IL = np.zeros((m, max_length_trg), dtype='uint32')  # <PAD> token is 0
    M = np.zeros((m, max_length_trg), dtype='uint32')
    for i, sent in enumerate(sentences):
        lengths.append(len(sent))
        for k, w in enumerate(sent):
            if k < max_length_trg:
                IL[i, k] = wtoi[w] if w in wtoi else wtoi['<UNK>']
                #M[i, k] = int(w in wtoi)
        # bar.update(i)
        if not i % 10000:
            print(".", end="")
    assert np.all(np.array(lengths) > 0), 'error: some line has no words'
    return IL, M, np.array(lengths)

# alphabet_size: alphabet size after special symbols appending
# Ts_max: max source sequence length
# Ts: this particular source sequence length
# labels: 1d tensor of (Tt_max) size
# ctc_blank_idx: CTC will treat this symbol as blank
# xy_ratio_max: the beam expanding rate
# trg_emb_pad_idx: embedding will represent this symbol as all-0-vector
# returns tensor of size (Ts_max, alphabet_size) ready for embedding of single sentence
def trg2alphabet(alphabet_size, Ts_max, Ts, Tt_max, labels, xy_ratio_max=7.0, trg_emb_pad_idx=1, ctc_blank_idx=0):
    #a = torch.arange(Ts_max*Tt_max).cuda().view(Ts_max, Tt_max)
    a = torch.arange(Ts_max*Tt_max).view(Ts_max, Tt_max)
    mask = ((a // Tt_max).int() <= ((a % Tt_max) * xy_ratio_max).int() + 3)
    mask *= ((a // Tt_max).int() >= ((a % Tt_max) * 1.0).int() - 2)
    labels2d = torch.from_numpy(labels.astype(int)).unsqueeze(0).repeat(Ts_max, 1)
    labels2d *= mask
    alphabet = torch.arange(alphabet_size, device='cpu')
    alphabet = alphabet.unsqueeze(0).repeat(Ts_max, 1)
    indexes = torch.unique(labels2d, sorted=False, dim=1)

    ix_rows = torch.arange(Ts_max, dtype=int, device='cpu').unsqueeze(1).repeat(1, indexes.size(1))
    ix_rows *= alphabet_size
    ix_rows = ix_rows.view_as(indexes)

    ix = indexes + ix_rows
    ix = torch.flatten(ix).unique()
    ab_mask = torch.zeros_like(alphabet)
    ab_mask = torch.flatten(ab_mask)
    ab_mask.index_fill_(0, ix, 1)
    ab_mask = ab_mask.view_as(alphabet)
    alphabet *= ab_mask
    #print((alphabet[:, :] > 0).sum())
    alphabet[alphabet == 0] = trg_emb_pad_idx
    alphabet[:, 0] = ctc_blank_idx
    alphabet[Ts:, :] = trg_emb_pad_idx
    return alphabet

def encode_sentences_train(lengths_train, sentences, params, wtoi):
    """
    encode all sentences into one large array, which will be 1-indexed.
    No special tokens are added, except from the <pad> after the effective length
    """
    alphabet_size = len(wtoi)
    max_xy_ratio = params.max_xy_ratio
    Tt_max = params.max_length_trg
    Ts_max = params.max_length_src
    lengths = []
    n_sentences = len(sentences)
    M = np.zeros(shape=(n_sentences, Ts_max, alphabet_size), dtype='uint32')
    IL = np.zeros((n_sentences, Tt_max), dtype='uint32')  # <PAD> token is 0
    indexes = np.arange(n_sentences, dtype=int)
    for (i, Ts, sent) in zip(indexes, lengths_train, sentences):
        Tt = len(sent)
        lengths.append(Tt)
        for k, w in enumerate(sent):
            if k < Tt_max:
                IL[i, k] = wtoi[w] if w in wtoi else wtoi['<UNK>']
        M[i] = trg2alphabet(alphabet_size, Ts_max, Ts, Tt_max, IL[i], max_xy_ratio)
        if (i % 10000) == 0:
            print(".", end="")
    assert np.all(np.array(lengths) > 0), 'error: some line has no words'
    return IL, M, np.array(lengths)

def main_trg(src_lengths_train, params):
    """
    Main preprocessing
    """
    select = """
        select	trim(regexp_replace(lower(af.pronounced_text), '[\\'' ]', '', 'g')) pronounced_text
        from 	audio_files af
                join audio_files_order fo on (fo.id = af.id and fo.subset = %s)
                join feature_counts fc on (fc.file_id = fo.id and fc.cnt_bulk between 40 and %s)
        order by fo.sort
        """
    max_length_src = params.max_length_src
    max_length_trg = params.max_length_trg

    print('...Creating vocabulary of the %d frequent tokens' % params.alphabet_size_trg)
    vocab_file = "data/%s/vocab.%s" % (params.data_dir, params.trg)
    vocab = build_vocab_from_db(params.alphabet_size_trg, vocab_file)
    print('...Vocabulary size:', len(vocab))
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    dsext = {'l': 'train', 'v': 'val', 't': 'test'}

    # create output h5 file
    f = h5py.File('data/%s/%s.h5' % (params.data_dir, params.trg), "w")

    with PgHelper() as pg_object:
        for dset in dsext:
            nodes = []
            n_rows = 0
            f.create_dataset("labels_" + dsext[dset], shape=(0,max_length_trg), dtype='uint32', chunks=True, maxshape=(None,max_length_trg))
            f.create_dataset("lengths_" + dsext[dset], shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
            if dset == 'l':
                f.create_dataset("masks_train", shape=(0,max_length_src,len(vocab)), dtype='uint32', chunks=True, maxshape=(None,max_length_src,len(vocab)))

            print('Loading targets for ' + dsext[dset] + ' .', end="")
            for row in pg_object.exec(select, (dset, max_length_src,)):
                sent = row[0].strip()[:max_length_trg]
                nodes.append(sent)
                n_rows += 1
                if (n_rows % 500) == 0:
                    sentences = np.array(nodes)
                    len_adv = len(sentences)
                    nodes = []
                    if dset == 'l':
                        Labels_trg, Mask_trg, Lengths_trg = encode_sentences_train(src_lengths_train, sentences, params, wtoi)
                        f["masks_train"].resize(f["masks_train"].shape[0] + len_adv, axis=0)
                        f["masks_train"][-len_adv:] = Mask_trg
                    else:
                        Labels_trg, _, Lengths_trg = encode_sentences(sentences, params, wtoi)
                    f["labels_" + dsext[dset]].resize(f["labels_" + dsext[dset]].shape[0] + len_adv, axis=0)
                    f["lengths_" + dsext[dset]].resize(f["lengths_" + dsext[dset]].shape[0] + len_adv, axis=0)
                    f["labels_" + dsext[dset]][-len_adv:] = Labels_trg
                    f["lengths_" + dsext[dset]][-len_adv:] = Lengths_trg
                    print('.', end="")

            if not (n_rows % 500) == 0:
                sentences = np.array(nodes)
                len_adv = len(sentences)
                nodes = []
                if dset == 'l':
                    Labels_trg, Mask_trg, Lengths_trg = encode_sentences_train(src_lengths_train, sentences, params, wtoi)
                    f["masks_train"].resize(f["masks_train"].shape[0] + len_adv, axis=0)
                    f["masks_train"][-len_adv:] = Mask_trg
                else:
                    Labels_trg, _, Lengths_trg = encode_sentences(sentences, params, wtoi)
                f["labels_" + dsext[dset]].resize(f["labels_" + dsext[dset]].shape[0] + len_adv, axis=0)
                f["lengths_" + dsext[dset]].resize(f["lengths_" + dsext[dset]].shape[0] + len_adv, axis=0)
                f["labels_" + dsext[dset]][-len_adv:] = Labels_trg
                f["lengths_" + dsext[dset]][-len_adv:] = Lengths_trg
                print('.', end="")
            print('\n')
            print("Read %d lines from data/%s/%s.%s" % (n_rows, params.data_dir, dsext[dset], params.trg))
    print('Wrote h5 file for the target data')
    f.flush()
    f.close()
    pdump({'itow': itow, 'params': params},
          'data/%s/%s.infos' % (params.data_dir, params.trg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='WMT14')
    parser.add_argument('--src', type=str, default='voice')
    parser.add_argument('--trg', type=str, default='ru')
    parser.add_argument('--max_xy_ratio', default=7.0, type=int,
                        help='the beam expanding rate for 2d(src, trg) embedding (train only)')
    parser.add_argument('--max_length_src', default=500, type=int,
                        help='max number of feature maps per sentence in the input')
    parser.add_argument('--alphabet_size_trg', default=30000, type=int,
                        help="Max words in the target vocabulary")
    parser.add_argument('--max_length_trg', default=50, type=int,
                        help='max length of a sentence in characters, no blanks')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size to sort by length')
    parser.add_argument('--use_centroids', action='store_true',
                        help='use centroid feature maps instead of bulk')
    params = parser.parse_args()

    if params.data_dir == 'm_ailabs':
        params.src = "voice"
        params.trg = "ru"
        params.max_length_src = 1000
        params.alphabet_size_trg = 100
        params.max_length_trg = 70
        params.batch_size = 16
        params.use_centroids = True        
    elif params.data_dir == 'm_ailabs_bulk':
        params.src = "voice"
        params.trg = "ru"
        params.max_length_src = 1000
        params.alphabet_size_trg = 100
        params.max_length_trg = 250
        params.batch_size = 16
        params.use_centroids = False

    print('Source language: ', params.src)
    h5file_name = 'data/%s/%s.h5' % (params.data_dir, params.src)
    # if osp.exists(h5file_name):
    #     os.remove(h5file_name)
    if osp.exists(h5file_name):
        f = h5py.File(h5file_name, "r")
    else:
        f = h5py.File(h5file_name, "w")
        select = """
            select 	array_cat(aa.feature_list, array_fill(0, array[%s-array_length(aa.feature_list, 1),20])::text[])::real[]
                    features_2d,
                    array_length(aa.feature_list, 1) nodes_count
            from (
                select 	fa.file_id, array_agg(fa.feature_values order by fa.feature_n) feature_list
                from 	(
                        select 	fe.file_id , fe.feature_n, string_to_array(fe.legend, '|') feature_values
                        from 	features fe
                        where	fe.set_name = %s
                        ) fa
                group by fa.file_id
                ) aa
                join audio_files_order fo on (fo.id = aa.file_id and fo.subset = %s)
                join feature_counts fc on (fc.file_id = fo.id and fc.cnt_bulk between 40 and %s)
            order by fo.sort 
            """
        max_length_src = params.max_length_src
        f.create_dataset("labels_train", shape=(0,max_length_src,20), dtype='float32', chunks=True, maxshape=(None,max_length_src,20))
        f.create_dataset("lengths_train", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        f.create_dataset("labels_val", shape=(0,max_length_src,20), dtype='float32', chunks=True, maxshape=(None,max_length_src,20))
        f.create_dataset("lengths_val", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        f.create_dataset("labels_test", shape=(0,max_length_src,20), dtype='float32', chunks=True, maxshape=(None,max_length_src,20))
        f.create_dataset("lengths_test", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        dsext = {'l': 'train', 'v': 'val', 't': 'test'}
        #dsext = {'v': 'val', 't': 'test'}
        set_name = 'b'
        if params.use_centroids:
            set_name = 'c'
        with PgHelper() as pg_object:
            #IL_val_src = np.array([row[0] for row in pg_object.exec(select)])
            for dset in dsext:
                nodes = []
                node_counts = []
                n_rows = 0
                print('Loading ' + dsext[dset] + ' .', end="")
                for row in pg_object.exec(select, (max_length_src, set_name, dset, max_length_src,)):
                    nodes.append(row[0])
                    node_counts.append(row[1])
                    n_rows += 1
                    if (n_rows % 500) == 0:
                        IL_src = np.array(nodes)
                        Lengths_src = np.array(node_counts)
                        nodes = []
                        node_counts = []
                        f["labels_" + dsext[dset]].resize(f["labels_" + dsext[dset]].shape[0] + IL_src.shape[0], axis=0)
                        f["lengths_" + dsext[dset]].resize(f["lengths_" + dsext[dset]].shape[0] + Lengths_src.shape[0], axis=0)
                        f["labels_" + dsext[dset]][-IL_src.shape[0]:] = IL_src
                        f["lengths_" + dsext[dset]][-Lengths_src.shape[0]:] = Lengths_src
                        print('.', end="")

                if not (n_rows % 500) == 0:
                    IL_src = np.array(nodes)
                    Lengths_src = np.array(node_counts)
                    nodes = []
                    node_counts = []
                    f["labels_" + dsext[dset]].resize(f["labels_" + dsext[dset]].shape[0] + IL_src.shape[0], axis=0)
                    f["lengths_" + dsext[dset]].resize(f["lengths_" + dsext[dset]].shape[0] + Lengths_src.shape[0], axis=0)
                    f["labels_" + dsext[dset]][-IL_src.shape[0]:] = IL_src
                    f["lengths_" + dsext[dset]][-Lengths_src.shape[0]:] = Lengths_src
                print('\n')
                print('Wrote h5 file for the source data')
        f.flush()

    #exit(0)
    pdump({'params': params},
          'data/%s/%s.infos' % (params.data_dir, params.src))

    print('\nTarget language: ', params.trg)
    main_trg(f["lengths_train"], params)

    f.close()
