import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from seq2seq import *
import sconce
import torch.optim
from utils import *
from preprocess import corpus
from deep_text_corrector import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

USE_CUDA = False

final_steps = 50000
plot_every = 200
print_every = 1
save_every = 1000
learning_rate = 0.0001
teacher_forcing_ratio = 0.5
clip = 5.0

job = sconce.Job('seq2seq-translate', {
    'attn_model': attn_model,
    'n_layers': n_layers,
    'dropout_p': dropout_p,
    'hidden_size': hidden_size,
    'learning_rate': learning_rate,
    'teacher_forcing_ratio': teacher_forcing_ratio,
})
job.plot_every = plot_every
job.log_every = print_every

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    batch_size, input_length = input_variable.size()
    batch_size, target_length = target_variable.size()

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables

    decoder_input = Variable(torch.LongTensor([[SOS_token] for _ in range(batch_size)]))
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    if random.random() < teacher_forcing_ratio:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[:, di])

            decoder_input = target_variable[:, di].unsqueeze(1)  # Next target is next input
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[:, di])

            # Get most likely word index (highest value) from output
            _, top_index = decoder_output.data.topk(1)
            ni = top_index
            # print ni, top_index.size()
            decoder_input = Variable(
                torch.LongTensor(ni))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            # TODO
            # if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


state = load_state()
step = 1
if state:
    step = state['step'] + 1
encoder, decoder = get_model(state=state)
encoder_optimizer, decoder_optimizer = get_optimizer(encoder, decoder, lr=learning_rate, state=state)
criterion = nn.NLLLoss()

for step in range(step, final_steps + 1):

    # Get training data for this cycle
    inputs, targets = corpus.next_batch()
    # n_inputs = map(lambda s: len(s), inputs)
    input_variable = Variable(torch.LongTensor(inputs))
    target_variable = Variable(torch.LongTensor(targets))

    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()

    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    job.record(step, loss)

    if step == 0: continue

    if step % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s: %s (%d %d%%) %.4f' % (step,
                                                   time_since(start, 1. * step / final_steps), step, step / final_steps * 100, print_loss_avg)
        print(print_summary)

    # if step % plot_every == 0:
    #     plot_loss_avg = plot_loss_total / plot_every
    #     plot_losses.append(plot_loss_avg)
    #     plot_loss_total = 0

    if step % save_every == 0:
        save_state(encoder, decoder, encoder_optimizer, decoder_optimizer, step)

# show_plot(plot_losses)