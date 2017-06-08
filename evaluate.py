import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from preprocess import *
from model import *
from model_state import *

USE_CUDA = False



def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                     decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly():
    pair = random.choice(pairs)

    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)

    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')


encoder, decoder = get_model()
evaluate_randomly()