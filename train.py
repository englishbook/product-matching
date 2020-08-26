# -*- coding: utf-8 -*-

import keras.backend.tensorflow_backend as ktf

from trainer.trainer import *
from data_process.data_loader import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
ktf.set_session(sess)


dir = 'your_bert_path'

bert_dir = os.path.join(dir, 'uncased_L-12_H-768_A-12')
bert_vocab_file = os.path.join(bert_dir, 'vocab.txt')
bert_config_file = os.path.join(bert_dir, 'bert_config.json')
bert_model_file = os.path.join(bert_dir, 'bert_model.ckpt')


def get_bert_data(train_file, valid_file, columns, bert_type=1, max_len=200):
    bert_ids_train, bert_segs_train, y_train, train_features, max_len = load_bert_data(train_file, columns=columns,
                                                                                       bert=bert_type,
                                                                                       bert_vocab_file=bert_vocab_file,
                                                                                       max_len=max_len)
    bert_ids_valid, bert_segs_valid, y_valid, valid_features, _ = load_bert_data(valid_file, columns=columns,
                                                                                 bert=bert_type,
                                                                                 bert_vocab_file=bert_vocab_file,
                                                                                 max_len=max_len)
    return bert_ids_train, bert_segs_train, y_train, train_features, bert_ids_valid, bert_segs_valid, y_valid, \
           valid_features, max_len


if __name__ == '__main__':
    model_name = ['bert_focal', 'bert_focal', 'bert_focal']
    computers = [False, False, True]
    columns = [['title'], ['title', 'description'], ['title', 'description']]

    use_bert = True
    bert_trainable = True
    feature = False
    swa = True
    fold = 5
    overwrite = False

    for i in range(len(model_name)):
        if computers[i]:
            train_file = 'data/computers_train_xlarge.json'
            valid_file = 'data/computers_gs.json'
        else:
            train_file = 'data/all_train_xlarge.json'
            valid_file = 'data/all_gs.json'

        if len(columns[i]) == 1:
            max_len = 64
        else:
            max_len = 200

        bert_ids_train, bert_segs_train, y_train, train_features, bert_ids_valid, bert_segs_valid, y_valid, \
        valid_features, max_len_bert = \
            get_bert_data(train_file, valid_file, columns[i], max_len=max_len)

        train = Trainer(model_name[i], use_bert, max_len_bert=max_len_bert,
                        bert_trainable=bert_trainable, bert_config_file=bert_config_file,
                        bert_model_file=bert_model_file,
                        feature=feature, swa=swa, columns=columns[i], computers=computers[i])

        train.fit(bert_ids_train, bert_segs_train, y_train, bert_ids_valid, bert_segs_valid, y_valid,
                  train_features=train_features, valid_features=valid_features, overwrite=overwrite)
