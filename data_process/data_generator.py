# -*- coding: utf-8 -*-

import numpy as np
import math
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data, label=None, batch_size=64):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.data_size = len(self.data[0])
        self.indices = np.arange(self.data_size)
        np.random.shuffle(self.indices)
        self.steps = int(math.ceil(self.data_size / self.batch_size))

    def __len__(self):
        return self.steps

    def on_epoch_begin(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data, batch_label = [], []
        for i in batch_index:
            d = []
            for j in range(len(self.data)):
                d.append(self.data[j][i])
            batch_data.append(d)
        batch_data = zip(*batch_data)
        batch_data = [np.asarray(d) for d in batch_data]
        if self.label is not None:
            batch_label = [self.label[i] for i in batch_index]
            return batch_data, np.asarray(batch_label)
        else:
            return batch_data
