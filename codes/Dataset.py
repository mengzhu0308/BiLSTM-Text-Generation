#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:51
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, texts, labels, tf=None):
        self.texts = texts
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        if self.tf is not None:
            text = self.tf(text)
            label = self.tf(label)

        return text, label