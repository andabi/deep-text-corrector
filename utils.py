# -*- coding: utf-8 -*-
#!/usr/bin/env python

import math
import time
import numpy as np


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    s = now() - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def now():
    return time.time()


# r = reference, h = hypothesis
def wer(r, h):
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(r))


# def show_attention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     show_plot_visdom()
#     plt.show()
#     plt.close()
#
#
# def evaluate_and_show_attention(input_sentence, target_sentence=None):
#     output_words, attentions = evaluate(input_sentence)
#     output_sentence = ' '.join(output_words)
#     print('>', input_sentence)
#     if target_sentence is not None:
#         print('=', target_sentence)
#     print('<', output_sentence)
#
#     show_attention(input_sentence, output_words, attentions)
#
#     # Show input, target, output text in visdom
#     win = 'evaluted (%s)' % hostname
#     text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
#     vis.text(text, win=win, opts={'title': win})
