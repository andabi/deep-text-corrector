import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from model import *
import sconce
import torch.optim
from utils import *
from preprocess import *
from model_state import *

USE_CUDA = False

final_steps = 50000
plot_every = 200
print_every = 100
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
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

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
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

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

    if step % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    if step % save_every == 0:
        save_state(encoder, decoder, encoder_optimizer, decoder_optimizer, step)

show_plot(plot_losses)