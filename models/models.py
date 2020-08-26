# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint as bert_load

from layers import *


class BaseModel(object):
    def __init__(self, config, use_bert, bert_type, max_len_bert, feature=False,
                 bert_trainable=False, bert_config_file=None, bert_checkpoint_file=None):
        self.config = config
        self.use_bert = use_bert
        self.bert_type = bert_type
        self.bert_trainable = bert_trainable
        self.bert_config_file = bert_config_file
        self.bert_checkpoint_file = bert_checkpoint_file
        self.bert_output_layer_num = self.config.bert_output_layer_num
        self.max_len_bert = max_len_bert
        self.feature = feature

    def build_inputs(self):
        model_inputs = []
        input_embed_a, input_embed_b, input_embed = [], [], None

        if self.use_bert and self.bert_type == 1:
            bert_model = bert_load(self.bert_config_file, self.bert_checkpoint_file,
                                   trainable=self.bert_trainable, output_layer_num=self.bert_output_layer_num,
                                   seq_len=self.max_len_bert)
            for l in bert_model.layers:
                l.trainable = True
            input_bert = Input(shape=(self.max_len_bert,))
            input_seg = Input(shape=(self.max_len_bert,))
            model_inputs.append(input_bert)
            model_inputs.append(input_seg)
            input_embed = NonMaskingLayer()(bert_model([input_bert, input_seg]))

        if self.use_bert and self.bert_type == 0:
            bert_model = bert_load(self.bert_config_file, self.bert_checkpoint_file,
                                   trainable=self.bert_trainable, output_layer_num=self.bert_output_layer_num,
                                   seq_len=self.max_len_bert)
            for l in bert_model.layers:
                l.trainable = True
            input_bert_a = Input(shape=(self.max_len_bert,))
            input_seg_a = Input(shape=(self.max_len_bert,))
            model_inputs.append(input_bert_a)
            model_inputs.append(input_seg_a)
            input_bert_b = Input(shape=(self.max_len_bert,))
            input_seg_b = Input(shape=(self.max_len_bert,))
            model_inputs.append(input_bert_b)
            model_inputs.append(input_seg_b)
            input_embed_a.append(NonMaskingLayer()(bert_model([input_bert_a, input_seg_a])))
            input_embed_b.append(NonMaskingLayer()(bert_model([input_bert_b, input_seg_b])))

        if self.feature:
            input_features = Input(shape=(1,), dtype='float32')
            model_inputs.append(input_features)

        if isinstance(input_embed_a, list) and len(input_embed_a) > 0:
            input_embed_a = concatenate(input_embed_a) if len(input_embed_a) > 1 \
                else input_embed_a[0]
            input_embed_b = concatenate(input_embed_b) if len(input_embed_b) > 1 \
                else input_embed_b[0]
        return model_inputs, input_embed_a, input_embed_b, input_embed

    def build_model(self):
        raise NotImplementedError


def binary_focal_loss(gamma=2, alpha=0.75):
    """
    Thanks for https://github.com/mkocabas/focal-loss-keras
    Binary form of focal loss.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


class BertModel(BaseModel):
    def __init__(self, config, use_bert, bert_type, max_len_bert,
                 bert_trainable=False, bert_config_file=None, bert_checkpoint_file=None,
                 optimizer=Adam(lr=2e-5), feature=False):
        self.optimizer = optimizer
        super(BertModel, self).__init__(config, use_bert, 1, max_len_bert,
                                        bert_trainable=bert_trainable, bert_config_file=bert_config_file,
                                        bert_checkpoint_file=bert_checkpoint_file,
                                        feature=feature)

    def build_model(self):
        model_inputs, a_embedding, b_embedding, pool_embedding = self.build_inputs()
        pool_embedding = Lambda(lambda x: x[:, 0])(pool_embedding)
        if self.feature:
            pool_embedding = concatenate([pool_embedding, model_inputs[-1]])

        pool_embedding = Dropout(0.2)(pool_embedding)

        sent_rep = Dense(400, activation='relu')(pool_embedding)
        sent_rep = Dense(100, activation='relu')(sent_rep)

        matching_score = Dense(1, activation='sigmoid', name='matching')(sent_rep)

        spm_loss = 'binary_crossentropy'
        spm_metrics = 'binary_accuracy'
        spm_model = Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class BertModelFocal(BaseModel):
    def __init__(self, config, use_bert, bert_type, max_len_bert,
                 bert_trainable=False, bert_config_file=None, bert_checkpoint_file=None,
                 optimizer=Adam(lr=2e-5), feature=False):
        self.optimizer = optimizer
        super(BertModelFocal, self).__init__(config, use_bert, 1, max_len_bert,
                                             bert_trainable=bert_trainable, bert_config_file=bert_config_file,
                                             bert_checkpoint_file=bert_checkpoint_file,
                                             feature=feature)

    def build_model(self):
        model_inputs, a_embedding, b_embedding, pool_embedding = self.build_inputs()
        pool_embedding = Lambda(lambda x: x[:, 0])(pool_embedding)
        if self.feature:
            pool_embedding = concatenate([pool_embedding, model_inputs[-1]])

        pool_embedding = Dropout(0.2)(pool_embedding)

        sent_rep = Dense(400, activation='relu')(pool_embedding)
        sent_rep = Dense(100, activation='relu')(sent_rep)

        matching_score = Dense(1, activation='sigmoid', name='matching')(sent_rep)

        spm_metrics = 'binary_accuracy'
        spm_model = Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=binary_focal_loss(alpha=.75, gamma=2), metrics=[spm_metrics])
        return spm_model
