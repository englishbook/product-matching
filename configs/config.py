# -*- coding: utf-8 -*-

from keras.optimizers import Adam


class Config(object):
    def __init__(self):
        self.checkpoint_dir = 'checkpoints'
        self.exp_name = None
        self.max_len = None
        self.vocab_len = None
        self.num_epochs = 100
        self.learning_rate = 2e-5
        self.optimizer = Adam
        self.batch_size = 128
        self.verbose_training = 1
        self.checkpoint_monitor = "val_f1"
        self.checkpoint_mode = "max"
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_verbose = True
        self.early_stopping_monitor = 'val_f1'
        self.early_stopping_patience = 10
        self.early_stopping_mode = 'max'
        self.max_len_word = 512
        self.max_len_bert = 512
        self.embedding_path = "data"
        self.embedding_file = 'WDC_100_dim.embeddings.npy'
        self.features_len = 0
        self.features = []
        self.bert_output_layer_num = 1
