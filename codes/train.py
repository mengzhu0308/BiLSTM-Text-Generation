#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/19 18:06
@File:          train.py
'''

import math
import random
import numpy as np

from keras.layers import *
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import Callback

from ToOneHot import ToOneHot
from Dataset import Dataset
from generator import generator

max_len = 60
step = 3
batch_size = 128

with open('nietzsche.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

sentences, next_chars = [], []
for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

num_examples = len(sentences)
chars = sorted(list(set(text)))
char_indices = {char: i for i, char in enumerate(chars)}
chars_len = len(chars)

def str2id(s):
    out_s = []
    for c in s:
        out_s.append(char_indices[c])
    return out_s

X = [str2id(sentence) for sentence in sentences]
Y = [char_indices[next_char] for next_char in next_chars]
X, Y = np.array(X), np.array(Y)

dataset = Dataset(X, Y, tf=ToOneHot(chars_len))
gen = generator(dataset, batch_size=128)

text_in = Input(shape=(max_len, chars_len), dtype='float32')
out = Bidirectional(CuDNNLSTM(64, return_sequences=True))(text_in)
out = Bidirectional(CuDNNLSTM(64))(out)
out = Dense(chars_len, activation='softmax')(out)
model = Model(text_in, out)
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

def sample(preds, temperature=1.):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size=1)
    return np.argmax(probas)

def evaluate(model):
    start_index = random.randint(0, len(text) - max_len - 1)
    generated_text = text[start_index: start_index + max_len]
    print('****************************************************')
    print(generated_text)
    print('****************************************************')

    out_s = ''.join(generated_text)
    for i in range(400):
        sampled = np.zeros((1, max_len, chars_len), dtype='float32')
        sampled[0, np.arange(max_len), str2id(generated_text)] = 1.

        preds = model.predict_on_batch(sampled)[0]
        next_index = sample(preds)
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
        out_s += next_char

    print('----------------------------------------------------')
    print(out_s)
    print('----------------------------------------------------')

class Evaluator(Callback):
    def __init__(self):
        super(Evaluator, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        evaluate(self.model)

evaluator = Evaluator()

model.fit_generator(
    gen,
    steps_per_epoch=math.ceil(num_examples / batch_size),
    epochs=60,
    callbacks=[evaluator]
)