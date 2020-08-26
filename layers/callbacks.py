# -*- coding: utf-8 -*-

from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error
import numpy as np
import os
from absl import logging
import math
from data_process.data_generator import DataGenerator


class PMMetric(Callback):
    def __init__(self, valid_data, valid_labels, bert_trainable=False, batch_size=32):
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.passed = 0
        self.lr = 0
        self.epoch = 1
        self.bert_trainable = bert_trainable
        self.batch_size = batch_size
        super(PMMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_probs = self.model.predict(self.valid_data, batch_size=self.batch_size)
        y_pred = [r >= 0.5 for r in pred_probs]
        y_true = self.valid_labels
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch, r, p, f1))


class SWA(Callback):
    """
    This callback implements a stochastic weight averaging (SWA) method with constant lr as
    presented in the paper: "Izmailov et al. Averaging Weights Leads to Wider Optima and Better
    Generalization" (https://arxiv.org/abs/1803.05407)

    Author's implementation: https://github.com/timgaripov/swa
    """
    def __init__(self, swa_model, checkpoint_dir, model_name='weights', swa_start=1):
        """

        Args:
            swa_model: the model that used to store the average of the weights once SWA begins
            checkpoint_dir: the directory where the model will be saved in
            model_name: the name of model we're training
            swa_start: the epoch when averaging begins. We generally pre-train the network for a
                       certain amount of epochs to start (swa_start > 1), as opposed to starting to
                       track the average from the very beginning.
        """
        super(SWA, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.swa_start = swa_start
        self.swa_model = swa_model

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.swa_n = 0
        # Note: I found deep copy of a model with customized layer would give errors
        # self.swa_model = copy.deepcopy(self.model)  # make a copy of the model we're training

        # Note: Something wired still happen even though i use keras.models.clone_model method,
        #       so I build swa_model outside this callback and pass it as an argument.
        #       It's not fancy, but the best I can do :)
        # see: https://github.com/keras-team/keras/issues/1765
        # self.swa_model = keras.models.clone_model(self.model)
        self.swa_model.set_weights(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        # update running average of parameters
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)

    def on_train_end(self, logs=None):
        logging.info('Logging Info - Saving SWA model checkpoint: %s_swa.hdf5' % self.model_name)
        weights_file = os.path.join(self.checkpoint_dir, '%s_swa.hdf5' % self.model_name)
        self.swa_model.save_weights(weights_file)
        logging.info('Logging Info - SWA model Saved')
