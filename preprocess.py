# -*- coding: utf-8 -*-

import random
import numpy as np

USE_CUDA = False

PAD_token = 0
SOS_token = 1
EOS_token = 2

class WordDict:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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
    def __init__(self, path='./data/corpus.txt', max_length=100):
        self.lines = open(path).read().strip().split('\n')
        self.pairs = [[s for s in l.split('\t')] for l in self.lines]
        self.dict = WordDict()
        self.max_length = max_length
        for pair in self.pairs:
            self.dict.add_indexes(pair[0])
            self.dict.add_indexes(pair[1])

    def indexes_from_sentence(self, sentence, max_length):
        indexes = [self.dict.word2index[word] for word in sentence.split(' ')]
        indexes.append(EOS_token)
        indexes.extend([PAD_token for _ in range(max_length - len(indexes))])
        return indexes

    def next_batch(self, batch_size=10):
        pairs = np.array(random.sample(self.pairs, batch_size))
        inputs = np.array([self.indexes_from_sentence(s, self.max_length) for s in pairs[:, 0]])
        targets = np.array([self.indexes_from_sentence(s, self.max_length) for s in pairs[:, 1]])
        return inputs, targets

corpus = Corpus()