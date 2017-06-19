# -*- coding: utf-8 -*-
#!/usr/bin/env python

# import sconce
import torch.optim
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from model import *
from preprocess import *
from utils import *

final_steps = 50000
plot_every = 200
print_every = 1
save_every = 500
learning_rate = 0.0001
teacher_forcing_ratio = 0.5
clip = 5.0

# job = sconce.Job('seq2seq-translate', {
#     'attn_model': attn_model,
#     'n_layers': n_layers,
#     'dropout_p': dropout_p,
#     'hidden_size': hidden_size,
#     'learning_rate': learning_rate,
#     'teacher_forcing_ratio': teacher_forcing_ratio,
# })
# job.plot_every = plot_every
# job.log_every = print_every

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


# outputs: (B, S, V)
# targets: (B, S, V)
# lengths: (B, 1)
def masked_cross_entropy(logits, targets, lengths):
    batch_size, seq_len, n_classes = logits.size()
    assert (batch_size, seq_len) == targets.size()

    # mask = Variable(torch.LongTensor([[1 for _ in range(l)] for l in lengths.data]))
    # mask = mask.resize_as(targets)
    mask = sequence_mask(sequence_length=lengths, max_len=targets.size(1))

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = targets.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*targets.size()) * mask.float()
    return losses.sum() / lengths.float().sum()


def train(input_batch, len_inputs, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Get size of input and target sentences
    # batch_size, input_length = input_batch.size()
    batch_size, target_length = target_batch.size()

    # TODO parameter를 paddingsequence로 받게끔 하고 아래는 삭제
    length_targets = Variable(torch.LongTensor(map(lambda s: len(s), target_batch))).cuda()

    # Run words through encoder
    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_batch, len_inputs, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for _ in range(batch_size)]))
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    decoder_outputs = Variable(torch.FloatTensor(batch_size, target_length, decoder.output_size).zero_())

    if Config.use_cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        decoder_outputs = decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    if random.random() < teacher_forcing_ratio:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            decoder_outputs[:, di] = decoder_output
            decoder_input = target_batch[:, di].unsqueeze(1)  # Next target is next input
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            decoder_outputs[:, di] = decoder_output
            # Get most likely word index (highest value) from output
            _, top_index = decoder_output.data.topk(1)
            decoder_input = Variable(top_index)  # Chosen word is next input
            if Config.use_cuda: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            # TODO
            # if ni == EOS_token: break

    loss = masked_cross_entropy(decoder_outputs, target_batch, length_targets)

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


train_corpus, _, word_dict = build_corpus()
state = load_state()
step = 1
if state:
    step = state['step'] + 1
encoder, decoder = get_model(word_dict.n_words, state=state)
encoder_optimizer, decoder_optimizer = get_optimizer(encoder, decoder, lr=learning_rate, state=state)
criterion = nn.NLLLoss()


for step in range(step, final_steps + 1):

    # Get training data for this cycle
    inputs, targets, len_inputs, len_targets = train_corpus.next_batch()
    input_variable = Variable(torch.LongTensor(inputs), requires_grad=False)
    target_variable = Variable(torch.LongTensor(targets), requires_grad=False)

    if Config.use_cuda:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()

    # Run the train function
    loss = train(input_variable, len_inputs, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    # job.record(step, loss)

    if step == 0: continue

    if step % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s: %s (%d %d%%) %.4f' % (step,
                                                   time_since(start, 1. * step / final_steps), step, step / final_steps * 100, print_loss_avg)
        print(print_summary)

        if step % save_every == 0:
            save_state(encoder, decoder, encoder_optimizer, decoder_optimizer, step)

            # if step % plot_every == 0:
    #     plot_loss_avg = plot_loss_total / plot_every
    #     plot_losses.append(plot_loss_avg)
    #     plot_loss_total = 0

# show_plot(plot_losses)