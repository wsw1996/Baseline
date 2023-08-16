import torch
import torch.nn as nn
from torch.nn import functional as F

#
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

class TrajGRU(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, obs_len,pred_len,input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(obs_len*hidden_size, pred_len*hidden_size)  # 全连接层
        self.linear2 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(b,s * h)
        x = self.linear1(x)   #b,pred*h
        x = x.reshape(b,self.pred_len,-1)
        x = self.linear2(x)
        x = x.view(self.pred_len,b,-1)
        return x








