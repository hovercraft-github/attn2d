#!/usr/bin/env python3

"""
Main per-processing script
"""

import os.path as osp
import argparse
import h5py
import numpy as np
from nmt.utils import pdump
from pghelper import PgHelper


def build_vocab_from_db(max_words, vocab_file):
    """
    Build vocabulary
    """
    # Count occurrences of the certain dictionary item
    counts = {}
    # Count occurrences of sentences with the certain number of dictionary items
    sent_lengths = {}

    select = """
        select di.dic_item, count(1) n_times
        from 
            (
            select  regexp_split_to_table(
                trim(regexp_replace(lower(af.pronounced_text), '[\\'']', '', 'g'))
                , '') dic_item
            from 	audio_files af
            ) di
        group by di.dic_item
        order by di.dic_item
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

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top dictionary items and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    vocab = [w for (c, w) in cw[:max_words]]
    bad_words = [w for (c, w) in cw[max_words:]]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    # print('sentence length distribution (count, number of words):')
    # sum_len = sum(sent_lengths.values())
    # for i in range(max_len+1):
        # print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len))

    # additional special UNK token we will use below to map infrequent words to
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
           "bad words": bad_words,
           "lengths": sent_lengths},
          vocab_file + ".stats")

    return vocab

def encode_sentences(sentences, params, wtoi):
    """
    encode all sentences into one large array, which will be 1-indexed.
    No special tokens are added, except from the <pad> after the effective length
    """
    max_length = params.max_length
    lengths = []
    m = len(sentences)
    IL = np.zeros((m, max_length), dtype='uint32')  # <PAD> token is 0
    M = np.zeros((m, max_length), dtype='uint32')
    print('...Encoding ', end="")
    for i, sent in enumerate(sentences):
        lengths.append(len(sent))
        for k, w in enumerate(sent):
            if k < max_length:
                IL[i, k] = wtoi[w] if w in wtoi else wtoi['<UNK>']
                M[i, k] = int(w in wtoi)
        # bar.update(i)
        if not i % 10000:
            print(".", end="")

    print("\n")
    assert np.all(np.array(lengths) > 0), 'error: some line has no words'
    return IL, M, lengths


def main_trg(params, train_order, val_order, test_order, vocab=None):
    """
    Main preprocessing
    """
    select = """
        select	trim(regexp_replace(lower(af.pronounced_text), '[\\'']', '', 'g')) pronounced_text
        from 	audio_files af
                join audio_files_order fo on (fo.id = af.id and fo.subset = %s)
                join feature_counts fc on (fc.file_id = fo.id and fc.cnt_bulk >= 40)
        order by fo.sort
        """
    max_length = params.max_length
    train_trg = 'data/%s/train.%s' % (params.data_dir, params.trg)
    val_trg = 'data/%s/valid.%s' % (params.data_dir, params.trg)
    test_trg = 'data/%s/test.%s' % (params.data_dir, params.trg)
    if not params.src == 'voice':
        with open(train_trg, 'r') as f:
            sentences = f.readlines()
            sentences = [sent.strip().split()[:max_length] for sent in sentences]
        print("Read %d lines from %s" % (len(sentences), train_trg))
    else:
        with PgHelper() as pg_object:
            sentences = [row[0].strip()[:max_length] for row in pg_object.exec(select, ('l',))]
        #sentences = [sent.strip()[:max_length] for sent in sentences]
    if train_order is not None:
        sentences = [sentences[k] for k in train_order]

    if vocab is None:
        vocab_file = "data/%s/vocab.%s" % (params.data_dir, params.trg)
        if osp.exists(vocab_file):
            print('...Reading vocabulary file (%s)' % vocab_file)
            vocab = []
            for line in open(vocab_file, 'r'):
                vocab.append(line.strip())
            # if '<BOS>' not in vocab:
            #     print('Inserting BOS')
            #     vocab.insert(0, "<BOS>")
            # if '<EOS>' not in vocab:
            #     print('Inserting EOS')
            #     vocab.insert(0, "<EOS>")
            if '<UNK>' not in vocab:
                print('Inserting UNK')
                vocab.insert(1, "<UNK>")
            if '<PAD>' not in vocab:
                print('Inserting PAD')
                vocab.insert(0, "<PAD>")
        else:
            print('...Creating vocabulary of the %d frequent tokens' % params.max_words_trg)
            vocab = build_vocab_from_db(params.max_words_trg,vocab_file)
    print('...Vocabulary size:', len(vocab))
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    # encode captions in large arrays, ready to ship to hdf5 file
    IL_train, Mask_train, Lengths_train = encode_sentences(sentences, params, wtoi)

    if not params.src == 'voice':
        with open(val_trg, 'r') as f:
            sentences = f.readlines()
            sentences = [sent.strip().split()[:max_length] for sent in sentences]
    else:
        with PgHelper() as pg_object:
            sentences = [row[0].strip()[:max_length] for row in pg_object.exec(select, ('v',))]
    if val_order is not None:
        sentences = [sentences[k] for k in val_order]

    print("Read %d lines from %s" % (len(sentences), val_trg))
    IL_val, Mask_val, Lengths_val = encode_sentences(sentences, params, wtoi)

    if not params.src == 'voice':
        with open(test_trg, 'r') as f:
            sentences = f.readlines()
            sentences = [sent.strip().split()[:max_length] for sent in sentences]
    else:
        with PgHelper() as pg_object:
            sentences = [row[0].strip()[:max_length] for row in pg_object.exec(select, ('t',))]
    if test_order is not None:
        sentences = [sentences[k] for k in test_order]

    print("Read %d lines from %s" % (len(sentences), test_trg))
    IL_test, Mask_test, Lengths_test = encode_sentences(sentences, params, wtoi)

    # create output h5 file
    f = h5py.File('data/%s/%s.h5' % (params.data_dir, params.trg), "w")
    f.create_dataset("labels_train", dtype='uint32', data=IL_train)
    f.create_dataset("lengths_train", dtype='uint32', data=Lengths_train)

    f.create_dataset("labels_val", dtype='uint32', data=IL_val)
    f.create_dataset("lengths_val", dtype='uint32', data=Lengths_val)

    f.create_dataset("labels_test", dtype='uint32', data=IL_test)
    f.create_dataset("lengths_test", dtype='uint32', data=Lengths_test)

    print('Wrote h5file for the target langauge')
    pdump({'itow': itow, 'params': params},
          'data/%s/%s.infos' % (params.data_dir, params.trg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='WMT14')
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--trg', type=str, default='fr')
    parser.add_argument('--max_words_src', default=30000, type=int,
                        help="Max words in the source vocabulary")
    parser.add_argument('--max_words_trg', default=30000, type=int,
                        help="Max words in the target vocabulary")
    parser.add_argument('--max_length', default=50, type=int,
                        help='max length of a sentence')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size to sort by length')
    parser.add_argument('--sort', action='store_true',
                        help='sort the training set by source sequence length')
    parser.add_argument('--share_vocab', action='store_true',
                        help='share the source and target vocab')
    parser.add_argument('--shuffle_sort', action='store_true',
                        help='sort the training set by source sequence length')
    parser.add_argument('--shuffle_sort_eval', action='store_true',
                        help='sort the training set by source sequence length')
    params = parser.parse_args()
    # Default settings for IWSLT DE-EN & WMT EN-DE:
    if params.data_dir == 'iwslt':
        params.src = "de"
        params.trg = "en"
        params.max_words_src = 14000
        params.max_words_trg = 14000
        params.shuffle_sort = True
        params.shuffle_sort_eval = True
        params.max_length = 200
        params.batch_size = 32

    if params.data_dir == 'm_ailabs':
        params.src = "voice"
        params.trg = "ru"
        params.max_words_src = 100
        params.max_words_trg = 100
        params.shuffle_sort = False
        params.shuffle_sort_eval = False
        params.max_length = 100
        params.batch_size = 32

    print('Source language: ', params.src)
    h5file_name = 'data/%s/%s.h5' % (params.data_dir, params.src)
    if not osp.exists(h5file_name):
        f = h5py.File(h5file_name, "w")
        select = """
            select 	array_cat(aa.feature_list, array_fill(0, array[400-array_length(aa.feature_list, 1),20])::text[])::real[]
                    features_2d,
                    array_length(aa.feature_list, 1) nodes_count
            from (
                select 	fa.file_id, array_agg(fa.feature_values order by fa.feature_n) feature_list
                from 	(
                        select 	fe.file_id , fe.feature_n, string_to_array(fe.legend, '|') feature_values
                        from 	features fe
                        where	fe.set_name = 'c'
                        ) fa
                group by fa.file_id
                ) aa
                join audio_files_order fo on (fo.id = aa.file_id and fo.subset = %s)
                join feature_counts fc on (fc.file_id = fo.id and fc.cnt_bulk >= 40)
            order by fo.sort 
            """
        f.create_dataset("labels_train", shape=(0,400,20), dtype='float32', chunks=True, maxshape=(None,400,20))
        f.create_dataset("lengths_train", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        f.create_dataset("labels_val", shape=(0,400,20), dtype='float32', chunks=True, maxshape=(None,400,20))
        f.create_dataset("lengths_val", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        f.create_dataset("labels_test", shape=(0,400,20), dtype='float32', chunks=True, maxshape=(None,400,20))
        f.create_dataset("lengths_test", shape=(0,), dtype='uint32', chunks=True, maxshape=(None,))
        dsext = {'l': 'train', 'v': 'val', 't': 'test'}
        #dsext = {'v': 'val', 't': 'test'}
        with PgHelper() as pg_object:
            #IL_val_src = np.array([row[0] for row in pg_object.exec(select)])
            for dset in dsext:
                nodes = []
                node_counts = []
                n_rows = 0
                print('Loading ' + dsext[dset] + ' .', end="")
                for row in pg_object.exec(select, (dset,)):
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
                print('Wrote h5 file for the source langauge')

    #exit(0)
    pdump({'params': params},
          'data/%s/%s.infos' % (params.data_dir, params.src))

    train_order, val_order, test_order = None, None, None
    print('\nTarget language: ', params.trg)
    main_trg(params, train_order, val_order, test_order)
