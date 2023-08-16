import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# class TrajLSTM(nn.Module):
#     def __init__(self,input_size,hidden_size,output_size,num_layers):
#         super(TrajLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
#         self.fc1 = nn.Linear(hidden_size,output_size)
#
#     def forward(self,x):
#         xa,_ = self.lstm(x)
#         s,b,h = xa.shape
#         xa = self.fc1(xa)
#         xa = xa.view(s,b,-1)
#
#         return xa
#
#
# class TrajGRU(nn.Module):
#     """
#         Parameters：
#         - input_size: feature size
#         - hidden_size: number of hidden units
#         - output_size: number of output
#         - num_layers: layers of LSTM to stack
#     """
#
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super().__init__()
#
#         self.gru = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
#         self.linear1 = nn.Linear(hidden_size, 16)  # 全连接层
#         self.linear2 = nn.Linear(16, output_size)  # 全连接层
#
#     def forward(self, _x):
#         x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
#         s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
#         x = x.view(s * b, h)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = x.view(s, b, -1)
#         return x



class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.embedding= nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = self.dropout(self.embedding(x))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,output_size,embedding_size,hidden_size,num_layers,dropout):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0)

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(self.embedding(x))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hidden size]
        # hidden = [n layers, batch size, hidden size]
        # cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell


class TrajSeq(nn.Module):
    def __init__(self, encoder, decoder,device,teaching_force):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teaching_force = teaching_force

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"



    def forward(self, x, y):
        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our argoverse motion forecasting dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :]

        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < self.teaching_force

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs

