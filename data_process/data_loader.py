# -*- coding: utf-8 -*-

import os
import codecs
import json
import logging
import pandas as pd
import numpy as np
import copy
import string
import pickle

from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.corpus import stopwords


stop_words_with_punct = copy.deepcopy(stopwords.words('english'))
stop_words = list(
    map(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)), stop_words_with_punct))


def tokenize(s):
    if isinstance(s, float):
        if s != s:
            return []
    s = str(s)
    s = s.replace('&amp;', "")
    s = s.replace('&reg;', "")
    s = s.replace('&quot;', "")
    s = s.replace('\t;', " ")
    s = s.replace('\n;', " ")
    return s.lower().translate(str.maketrans('', '', string.punctuation)).split()


def process_text(s, join_word=True):
    if isinstance(s, float):
        if s != s:
            return s
    w_list = tokenize(s)
    w_clean_list = [x for x in w_list if x not in stop_words]
    if join_word:
        string_final = " ".join(w_clean_list)
        return string_final
    else:
        return w_clean_list


def get_len_from_corpus(corpus, corpus2=None):
    if corpus2 is None:
        lengths = [len(seq) for seq in corpus]
    else:
        lengths = [len(corpus[i] + corpus2[i]) for i in range(len(corpus))]
    l = sorted(lengths)[int(0.95 * len(corpus))]
    print('max_len:', l)
    return l


def load_bert_data(file, columns='title', bert=0, bert_vocab_file=None, max_len=None, test=False):
    from keras_bert import Tokenizer
    bert_vocab = {}
    with codecs.open(bert_vocab_file, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            bert_vocab[token] = len(bert_vocab)
    bert_tokenizer = Tokenizer(bert_vocab)

    input_left, input_right = [], []
    labels = []
    features = []
    pair_ids = []

    with codecs.open(file, 'r', encoding='utf-8') as f:
        for line_id, line in tqdm(enumerate(f)):
            data = json.loads(line)
            left, right = None, None
            if isinstance(columns, str):
                columns = [columns]
            for i, col in enumerate(columns):
                if data[col + '_left'] is None:
                    data[col + '_left'] = ''
                if data[col + '_right'] is None:
                    data[col + '_right'] = ''
                if i == 0:
                    left = data[col + '_left']
                    right = data[col + '_right']
                else:
                    left += ' ' + data[col + '_left']
                    right += ' ' + data[col + '_right']
            input_left.append(process_text(left, join_word=False))
            input_right.append(process_text(right, join_word=False))
            if not test:
                labels.append(float(data['label']))
            else:
                pair_ids.append([data['id_left'], data['id_right']])
            f = []
            if data['category_left'] == data['category_right']:
                f.append(1)
            else:
                f.append(0)
            features.append(f)

    if max_len is None and bert == 0:
        max_len = min(512, get_len_from_corpus(input_left + input_right) + 2)
    elif max_len is None and bert == 1:
        max_len = min(512, get_len_from_corpus(input_left, input_right) + 3)

    if bert == 0:
        bert_ids_a, bert_ids_b = [], []
        bert_seq_a, bert_seq_b = [], []
        for i in tqdm(range(len(input_right))):
            indices, segments = bert_tokenizer.encode(first=' '.join(input_left[i]),
                                                      max_len=max_len)
            bert_ids_a.append(indices)
            bert_seq_a.append(segments)
            indices, segments = bert_tokenizer.encode(first=' '.join(input_right[i]),
                                                      max_len=max_len)
            bert_ids_b.append(indices)
            bert_seq_b.append(segments)
        if not test:
            return (bert_ids_a, bert_ids_b), (bert_seq_a, bert_seq_b), np.asarray(labels), np.asarray(features), max_len
        return (bert_ids_a, bert_ids_b), (bert_seq_a, bert_seq_b), np.asarray(features), pair_ids, max_len
    else:
        bert_ids, bert_seg = [], []
        for i in tqdm(range(len(input_left))):
            indices, segments = bert_tokenizer.encode(first=' '.join(input_left[i]),
                                                      second=' '.join(input_right[i]),
                                                      max_len=max_len)
            bert_ids.append(indices)
            bert_seg.append(segments)
        if not test:
            return bert_ids, bert_seg, np.asarray(labels), np.asarray(features), max_len
        return bert_ids, bert_seg, np.asarray(features), pair_ids, max_len


def get_label(label_file, pair_ids):
    labels = {}
    with codecs.open(label_file, 'r', encoding='utf-8') as f:
        for line_id, line in tqdm(enumerate(f)):
            data = json.loads(line)
            labels[str(data['id_left']) + str(data['id_right'])] = int(data['label'])

    new_labels = []
    for pid in pair_ids:
        new_labels.append(labels[str(pid[0]) + str(pid[1])])
    return new_labels
