# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
import os

import torch.optim as optim

from seq2seq.seq2seq import *


def save_state(encoder, decoder, encoder_optim, decoder_optim, step, path='checkpoints/model'):
    state = {'step': step,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optim': encoder_optim.state_dict(),
             'decoder_optim': decoder_optim.state_dict()}
    filename = path + '-' + str(step)
    torch.save(state, filename)


def load_state(step=None, path='checkpoints/model'):
    state = {}
    file_list = glob.glob(path + '*')
    if file_list:
        if step:
            filename = path + '-' + str(step)
        else:
            filename = max(file_list, key=os.path.getctime)

        state = torch.load(filename)
    return state


def get_model(n_classes, state=None, step=None, load=True):
    encoder = EncoderRNN(n_classes, hidden_size, n_layers)
    decoder = AttnDecoderRNN(attn_model, hidden_size, n_classes, n_layers, dropout_p=dropout_p)
    if Config.use_cuda:
        encoder.cuda()
        decoder.cuda()

    if load:
        if not state:
            state = load_state(step)
        if state:
            encoder.load_state_dict(state['encoder'])
            decoder.load_state_dict(state['decoder'])

    return encoder, decoder


def get_optimizer(encoder, decoder, step=None, state=None, lr=0.0001):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    if not state:
        state = load_state(step)
    if state:
        encoder_optimizer.load_state_dict(state['encoder_optim'])
        decoder_optimizer.load_state_dict(state['decoder_optim'])

    return encoder_optimizer, decoder_optimizer