# -*- coding: utf-8 -*-

import os
import codecs
import json

from sklearn.model_selection import KFold

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from configs.config import Config
from models.models import *
from data_process.data_generator import DataGenerator


class Trainer(object):
    def __init__(self, model_name, use_bert, bert_type=1, max_len_bert=None,
                 bert_trainable=False, bert_config_file=None, bert_model_file=None,
                 feature=False, swa=True, seed=42, columns='title', computers=False):
        self.config = Config()
        self.model_name = model_name
        self.use_bert = use_bert
        self.bert_type = bert_type
        self.bert_trainable = bert_trainable
        self.feature = feature
        self.store_name = model_name
        if self.use_bert:
            if max_len_bert > 100:
                self.config.batch_size = 16
            else:
                self.config.batch_size = 32
            self.store_name += '_bert%d' % self.bert_type
            if not bert_trainable:
                self.store_name += '_fix'

        if self.feature:
            self.store_name += '_f'
        if computers:
            self.store_name += '_computers'

        self.seed = seed
        if isinstance(columns, list):
            columns = '_'.join(columns)
        self.columns = columns
        self.config.checkpoint_dir = os.path.join(self.config.checkpoint_dir, columns)
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        self.store_name = os.path.join(self.config.checkpoint_dir, self.store_name)
        print(self.store_name)
        if not os.path.exists(self.store_name):
            os.mkdir(self.store_name)
        self.config.max_len_bert = max_len_bert
        if bert_trainable:
            self.optimizer = Adam(lr=2e-5)
        else:
            self.optimizer = 'adam'
        self.bert_config_file = bert_config_file
        self.bert_model_file = bert_model_file
        self.callbacks = []
        self.swa = swa

    def init_callbacks(self, swa_model=None, weights_name='weights'):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.store_name, '%s.hdf5' % weights_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )

        if self.swa:
            self.callbacks.append(SWA(swa_model=swa_model, checkpoint_dir=self.store_name,
                                      model_name=weights_name, swa_start=5))

    def compile(self):
        all_model = {'bert': BertModel,
                     'bert_focal': BertModelFocal}
        self.model = all_model[self.model_name](self.config, self.use_bert,
                                                self.bert_type, self.config.max_len_bert,
                                                bert_trainable=self.bert_trainable,
                                                bert_config_file=self.bert_config_file,
                                                bert_checkpoint_file=self.bert_model_file,
                                                optimizer=self.optimizer,
                                                feature=self.feature).build_model()
        if self.swa:
            self.swa_model = all_model[self.model_name](self.config, self.use_bert,
                                                        self.bert_type, self.config.max_len_bert,
                                                        bert_trainable=self.bert_trainable,
                                                        bert_config_file=self.bert_config_file,
                                                        bert_checkpoint_file=self.bert_model_file,
                                                        optimizer=self.optimizer,
                                                        feature=self.feature).build_model()

    def pad(self, x_data_a, x_data_b, max_len):
        return pad_sequences(x_data_a, maxlen=max_len, padding='post', truncating='post'), \
               pad_sequences(x_data_b, maxlen=max_len, padding='post', truncating='post')

    def fit(self, bert_ids_train, bert_segs_train, y_train, bert_ids_valid, bert_segs_valid, y_valid,
            train_features=None, valid_features=None, overwrite=False):
        inputs_train, inputs_valid, self.inputs_test = [], [], []

        if self.use_bert and self.bert_type == 0:
            inputs_train += [np.asarray(bert_ids_train[0]), np.asarray(bert_segs_train[0]),
                             np.asarray(bert_ids_train[1]), np.asarray(bert_segs_train[1])]
            inputs_valid += [np.asarray(bert_ids_valid[0]), np.asarray(bert_segs_valid[0]),
                                 np.asarray(bert_ids_valid[1]), np.asarray(bert_segs_valid[1])]

        if self.use_bert and self.bert_type == 1:
            inputs_train += [np.asarray(bert_ids_train), np.asarray(bert_segs_train)]
            inputs_valid += [np.asarray(bert_ids_valid), np.asarray(bert_segs_valid)]

        if self.feature:
            inputs_train += [train_features]
            inputs_valid += [valid_features]

        self.compile()
        if not os.path.exists(os.path.join(self.store_name, 'weights.hdf5')) or overwrite:

            self.callbacks = []
            self.callbacks.append(PMMetric(inputs_valid, y_valid, bert_trainable=self.bert_trainable,
                                           batch_size=self.config.batch_size))
            if self.swa:
                self.init_callbacks(swa_model=self.swa_model, weights_name='weights')
            else:
                self.init_callbacks(weights_name='weights')

            train_generator = DataGenerator(inputs_train, y_train,
                                            batch_size=self.config.batch_size)
            valid_generator = DataGenerator(inputs_valid, y_valid,
                                            batch_size=self.config.batch_size)

            self.model.fit_generator(train_generator, epochs=self.config.num_epochs,
                                     verbose=self.config.verbose_training,
                                     validation_data=valid_generator,
                                     callbacks=self.callbacks)

        self.load_model_weights(os.path.join(self.store_name, 'weights.hdf5'))
        pred = self.get_predict(inputs_valid)
        self.evaluate(pred, y_valid)
        pred = self.check_cate(pred, valid_features)
        self.evaluate(pred, y_valid)

        if self.swa:
            self.load_model_weights(os.path.join(self.store_name, 'weights_swa.hdf5'))
            pred = self.get_predict(inputs_valid)
            print('swa...')
            self.evaluate(pred, y_valid)
            pred = self.check_cate(pred, valid_features)
            self.evaluate(pred, y_valid)

    def predict(self, bert_ids_test, bert_segs_test, test_features=None):
        model_path, model_name = os.path.split(self.store_name)
        result_dir = 'results'
        if not os.path.exists(os.path.join(result_dir, self.columns, model_name)):
            os.makedirs(os.path.join(result_dir, self.columns, model_name))

        inputs_test = []

        if self.use_bert and self.bert_type == 0:
            inputs_test += [np.asarray(bert_ids_test[0]), np.asarray(bert_segs_test[0]),
                             np.asarray(bert_ids_test[1]), np.asarray(bert_segs_test[1])]

        if self.use_bert and self.bert_type == 1:
            inputs_test += [np.asarray(bert_ids_test), np.asarray(bert_segs_test)]

        if self.feature:
            inputs_test += [test_features]

        self.compile()
        self.load_model_weights(os.path.join(self.store_name, 'weights_swa.hdf5'))
        y_pred = self.get_predict(inputs_test)
        y_pred = self.check_cate(y_pred, test_features)
        return y_pred

    def load_model_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def get_predict(self, data):
        y_pred = self.model.predict(data, batch_size=self.config.batch_size, verbose=1)
        return np.reshape(y_pred, (-1))

    def evaluate(self, results, labels):
        y_pred = [r >= 0.5 for r in results]
        r = recall_score(labels, y_pred)
        p = precision_score(labels, y_pred)
        f1 = f1_score(labels, y_pred)
        acc = accuracy_score(labels, y_pred)
        print('Recall: {}, Precision: {}, Acc: {}, F1: {}'.format(r, p, acc, f1))
        return f1

    def check_cate(self, pred, features):
        for i in range(len(pred)):
            if pred[i] >= 0.5 and features[i][0] == 0:
                pred[i] = 0
        return pred

    def write_prob(self, y_pred, test_pairs):
        with codecs.open(os.path.join(self.store_name, 'results.csv'), 'w', encoding='utf-8') as f:
            for i in range(len(test_pairs)):
                f.write(str(test_pairs[i][0]) + ',' + str(test_pairs[i][1]) + ',' + str(float(y_pred[i])) + '\n')

    def write_results(self, y_pred, test_pairs, filename):
        pred = [r >= 0.5 for r in y_pred]
        with codecs.open(filename, 'w', encoding='utf-8') as f:
            for i in range(len(test_pairs)):
                f.write(str(test_pairs[i][0]) + ',' + str(test_pairs[i][1]) + ',' + str(int(pred[i])) + '\n')

