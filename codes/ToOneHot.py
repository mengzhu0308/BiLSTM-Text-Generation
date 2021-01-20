#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/15 20:27
@File:          ToOneHot.py
'''

import numpy as np

class ToOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        data_size = data.size
        if data_size > 1:
            one_hot = np.zeros((data_size, self.num_classes), dtype='float32')
            one_hot[np.arange(data_size), data] = 1.
        else:
            one_hot = np.zeros(self.num_classes, dtype='float32')
            one_hot[data] = 1.

        return one_hot