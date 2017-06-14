import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = False

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

    def init_hidden(self, batch_size=10):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def forward(self, input_seq, hidden):
        # input_seq.size() = (B, S), hidden.size() = (L, B, H), embedded.size() = (B, S, H), output.size() = (B, S, H)
        # batch_size, seq_len = input_sequence.size()
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input, last_context, last_hidden, encoder_outputs):
        # input.size() = (B, 1), last_context.size() = (B, H), last_hidden.size() = (L, B, H), encoder_outputs.size() = (B, S, H)
        # word_embedded.size() = (B, 1, H)
        # print input.size()
        word_embedded = self.embedding(input)

        # rnn_input.size() = (B, 1, 2H), rnn_output.size() = (B, 1, H)
        # print word_embedded.size(), last_context.unsqueeze(1).size()
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(1)), -1)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        rnn_output = rnn_output.squeeze(1)  # B x S=1 x H -> B x H

        # atten_weights.size() = (B, S)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # B x H

        # TODO tanh?
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        output = self.out(torch.cat((rnn_output, context), -1))  # B x V

        # Return final output, hidden state, and attention weights (for visualization)
        # output.size() = (B, V)
        return output, context, hidden, attn_weights


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        # elif self.method == 'concat':
        #     self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        #     self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden.size() = (B, H), encoder_outputs.size() = (B, S, H)
        batch_size, encoder_outputs_len, _ = encoder_outputs.size()

        # Create variable to store attention energies
        # attn_energies.size() = (B, S)
        attn_energies = Variable(torch.zeros((batch_size, encoder_outputs_len)))  # B x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        # attn_energies.size() = (B, S)
        for i in range(encoder_outputs_len):
            attn_energies[:, i] = self.score(hidden, encoder_outputs[:, i])
            # print attn_energies[:, i]

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):

        # print hidden.size(), encoder_output.size()
        if self.method == 'dot':
            energy = hidden.unsqueeze(1).bmm(encoder_output.unsqueeze(2))  # dot product
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.unsqueeze(1).bmm(energy.unsqueeze(2))
            return energy

        # TODO
        # elif self.method == 'concat':
        #     energy = self.attn(torch.cat((hidden, encoder_output), -1))
        #     energy = self.other.unsqueeze(1).bmm(energy.unsqueeze(2))
        #     return energy


# class BahdanauAttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#
#         # Define parameters
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         # Define layers
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.dropout = nn.Dropout(dropout_p)
#         self.attn = GeneralAttn(hidden_size)
#         self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
#         self.out = nn.Linear(hidden_size, output_size)
#
#     def forward(self, word_input, last_hidden, encoder_outputs):
#         # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
#
#         # Get the embedding of the current input word (last output word)
#         word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
#         word_embedded = self.dropout(word_embedded)
#
#         # Calculate attention weights and apply to encoder outputs
#         attn_weights = self.attn(last_hidden[-1], encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
#
#         # Combine embedded input word and attended context, run through RNN
#         rnn_input = torch.cat((word_embedded, context), 2)
#         output, hidden = self.gru(rnn_input, last_hidden)
#
#         # Final output layer
#         output = output.squeeze(0)  # B x N
#         output = F.log_softmax(self.out(torch.cat((output, context), 1)))
#
#         # Return final output, hidden state, and attention weights (for visualization)
#         return output, hidden, attn_weights