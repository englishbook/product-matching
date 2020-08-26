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


def get_bert_data_test(test_file, columns, bert_type=1, max_len=200):
    bert_ids_test, bert_segs_test, test_features, pair_ids, max_len = load_bert_data(test_file, columns=columns,
                                                                                     bert=bert_type,
                                                                                     bert_vocab_file=bert_vocab_file,
                                                                                     max_len=max_len, test=True)
    return bert_ids_test, bert_segs_test, test_features, pair_ids, max_len


if __name__ == '__main__':
    model_name = ['bert_focal', 'bert_focal', 'bert_focal']
    computers = [False, False, True]
    columns = [['title'], ['title', 'description'], ['title', 'description']]

    test_file = 'data/testset_1500.json'
    test_label_file = 'data/task1_testset_1500_with_labels.json'
    use_bert = True
    bert_trainable = True
    feature = False
    swa = True
    fold = 5
    overwrite = False

    y_pred = None
    train = None
    pair_ids = None

    for i in range(len(model_name)):
        print('predict model %d' % (i + 1))
        if len(columns[i]) == 1:
            max_len = 64
        else:
            max_len = 200

        bert_ids_test, bert_segs_test, test_features, pair_ids, max_len_bert = \
            get_bert_data_test(test_file, columns[i], max_len=max_len)

        train = Trainer(model_name[i], use_bert, max_len_bert=max_len_bert,
                        bert_trainable=bert_trainable, bert_config_file=bert_config_file, bert_model_file=bert_model_file,
                        feature=feature, swa=swa, columns=columns[i], computers=computers[i])

        pred = train.predict(bert_ids_test, bert_segs_test, test_features=test_features)

        if y_pred is None:
            y_pred = pred
        else:
            y_pred += pred

    y_pred /= len(model_name)
    labels = get_label(test_label_file, pair_ids)
    train.evaluate(y_pred, labels)
    train.write_results(y_pred, pair_ids, 'results.csv')
