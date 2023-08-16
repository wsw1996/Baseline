import torch
import torch.nn as nn
from torch.nn import functional as F


# class TrajLSTM(nn.Module):
#     def __init__(self,obs_len,pred_len,input_size,hidden_size,output_size,num_layers):
#         super(TrajLSTM, self).__init__()
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
#         self.fc1 = nn.Linear(obs_len*hidden_size,pred_len*hidden_size)
#         self.fc2 = nn.Linear(hidden_size,output_size)
#
#     def forward(self,x):
#         xa,_ = self.lstm(x)
#         seq,b,h = xa.shape
#         xa = xa.view(b,seq * h)
#         xa = self.fc1(xa)
#         xa = xa.reshape(b,self.pred_len,-1)
#         xa = self.fc2(xa)
#         xa = xa.view(self.pred_len,b,-1)
#
#         return xa

class TrajLSTM(nn.Module):
    def __init__(self,obs_len,pred_len,input_size,hidden_size,output_size,num_layers):
        super(TrajLSTM, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc1 = nn.Linear(obs_len*hidden_size,pred_len*hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        xa,_ = self.lstm(x)
        seq,b,h = xa.shape
        xa = xa.view(b,seq * h)
        xa = self.fc1(xa)
        xa = xa.view(b,self.pred_len,-1)
        xa = self.fc2(xa)
        xa = xa.view(self.pred_len,b,-1)

        return xa


