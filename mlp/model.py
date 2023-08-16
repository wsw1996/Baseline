import torch
import torch.nn as nn
from torch.nn import functional as F
import random

class TrajLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(TrajLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc1 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        xa,_ = self.lstm(x)
        s,b,h = xa.shape
        xa = self.fc1(xa)
        xa = xa.view(s,b,-1)

        return xa


class TrajGRU(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, 16)  # 全连接层
        self.linear2 = nn.Linear(16, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        return x



class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x)))
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
        embedded = self.dropout(F.relu(self.embedding(x)))

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
    def __init__(self, encoder, decoder,device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"



    def forward(self, x, y, teacher_forcing_ratio=0.5):
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
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs

# from torch import nn
# import torch
# import torch.nn as nn
# from torch.nn.utils import weight_norm
#
#
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         """
#         其实这就是一个裁剪的模块，裁剪多出来的padding
#         """
#         return x[:, :, :-self.chomp_size].contiguous()
#
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         """
#         相当于一个Residual block
#
#         :param n_inputs: int, 输入通道数
#         :param n_outputs: int, 输出通道数
#         :param kernel_size: int, 卷积核尺寸
#         :param stride: int, 步长，一般为1
#         :param dilation: int, 膨胀系数
#         :param padding: int, 填充系数
#         :param dropout: float, dropout比率
#         """
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
#         self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#
#     def init_weights(self):
#         """
#         参数初始化
#
#         :return:
#         """
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         """
#         :param x: size of (Batch, input_channel, seq_len)
#         :return:
#         """
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
#
#
# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         """
#         TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
#         对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
#         对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。
#
#         :param num_inputs: int， 输入通道数
#         :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
#         :param kernel_size: int, 卷积核尺寸
#         :param dropout: float, drop_out比率
#         """
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
#             in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
#             out_channels = num_channels[i]  # 确定每一层的输出通道数
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
#
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         """
#         输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
#         这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
#         很巧妙的设计。
#
#         :param x: size of (Batch, input_channel, seq_len)
#         :return: size of (Batch, output_channel, seq_len)
#         """
#         return self.network(x)
#
#
# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size)
#         self.init_weights()
#
#     def init_weights(self):
#         self.linear.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         y1 = self.tcn(x)
#         return self.linear(y1[:, :, -1])
#
# from torch.nn.utils import weight_norm
# class delect_padding(nn.Module):
#
#     def __init__(self, chomp_size):
#         super(delect_padding, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         """
#         其实就是一个裁剪的模块，裁剪多出来的padding
#         """
#         return x[:, :, :, :-self.chomp_size].contiguous()
#
#
# class TCN(nn.Module):
#     # 输入数据(B,C1,H,W)
#     # 输出数据(B,C2,H,W-1)
#     # 通过调整padding值的设定可以改变输出数据的形式，在这里我设定的kernel_size=3，padding=1，W纬度得到W-1的输出
#     def __init__(self, n_inputs, n_outputs, kernel_size, padding, dropout=0.2):
#         """
#         :param n_inputs: int, 输入通道数
#         :param n_outputs: int, 输出通道数
#         :param kernel_size: int, 卷积核尺寸
#         :param padding: int, 填充系数
#         :param dropout: float, dropout比率
#         """
#         super(TCN, self).__init__()
#
#         self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size), padding=(0, padding)))
#         self.delect_pad = delect_padding(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.net = nn.Sequential(self.conv1, self.delect_pad, self.relu1, self.dropout1)
#         self.relu = nn.ReLU()
#         self.init_weights()
#
#     def init_weights(self):
#         """
#         参数初始化
#         """
#         self.conv1.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         return self.relu(out)

# a = torch.tensor((1,8,4,4))
# model = TCN( 8,12, (3), (1))
# b = model(a)
# print(b,b.shape)
# #这个可以实现特征向量4个变成2个
#
# class TrajCNN(nn.Module):
#     def __init__(self, obs_len,pred_len,hidden_size,kernel_size):
#         super(TrajCNN, self).__init__()
#
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=obs_len, out_channels=hidden_size, kernel_size=kernel_size),  # 24 - 2 + 1 = 23
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=kernel_size, stride=1),  # 23 - 2 + 1 = 22
#         )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=kernel_size),  # 22 - 2 + 1 = 21
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=kernel_size, stride=1),  # 21 - 2 + 1 = 20
#         # )
#         self.Linear1 = nn.Linear(hidden_size, pred_len * 20)
#         self.Linear2 = nn.Linear(pred_len * 20, pred_len)
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         x = self.conv1(x)
#         x = x.permute(0,2,1)   #(N,hidden,2)--->(N,2,hidden)
#         x = self.Linear1(x)
#         x = self.relu(x)
#         x = self.Linear2(x)      #(N,2,pred_len)
#         x = x.permute(2, 0, 1)     #(序列长度，batch,output-size)
#         return x

class TrajLinear(nn.Module):
    def __init__(self, obs_len,pred_len,hidden_size,input_size,output_size):
        super(TrajLinear, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(obs_len*hidden_size, pred_len*hidden_size)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.Linear1(x)     #(seq_len,batch,hidden)
        x = self.relu(x)
        x = x.permute(1, 0, 2)
        b = x.shape[0]
        x = x.reshape(b,-1)
        x = self.Linear2(x)
        x = self.relu(x)     #（batch,pred_len*hidden）
        x = x.reshape(b,self.pred_len,-1)
        x = self.Linear3(x)  #(batch，pred_len,output-size)
        x = x.permute(1, 0, 2)

        return x

    #linear相当于三层linear