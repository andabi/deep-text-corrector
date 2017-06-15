# -*- coding: utf-8 -*-

import random
import numpy as np
from config import Config

USE_CUDA = True

PAD_token = 0
SOS_token = 1
EOS_token = 2


class WordDict:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.n_words = 3  # Count PAD, SOS and EOS

    def add_indexes(self, sentence):
        for word in sentence.split(' '):
            self.add_index(word)

    def add_index(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Corpus:
    def __init__(self, max_length, path='./data/corpus.txt'):
        self.lines = open(path).read().strip().split('\n')
        self.pairs = [[s for s in l.split('\t')] for l in self.lines]
        self.dict = WordDict()
        self.max_length = max_length
        for pair in self.pairs:
            self.dict.add_indexes(pair[0])
            self.dict.add_indexes(pair[1])

    def sentence_to_indexes(self, sentence, max_length):
        indexes = [self.dict.word2index[word] for word in sentence.split(' ')]
        indexes.append(EOS_token)
        indexes.extend([PAD_token for _ in range(max_length - len(indexes))])
        return indexes

    def indexes_to_sentence(self, indexes):
        indexes = filter(lambda i: i != PAD_token, indexes)
        indexes = map(lambda i: self.dict.index2word[i], indexes)
        return ' '.join(indexes)

    def next_batch(self, batch_size=100):
        pairs = np.array(random.sample(self.pairs, batch_size))
        # inputs, targets = pairs[:, 0], pairs[:, 1]
        input_indexes = [self.sentence_to_indexes(s, self.max_length) for s in pairs[:, 0]]
        targets_indexes = [self.sentence_to_indexes(s, self.max_length) for s in pairs[:, 1]]
        return input_indexes, targets_indexes

corpus = Corpus(Config.max_seq_length)