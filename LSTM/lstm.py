import torch
from torch import nn


class LSTMModel(nn.Module):
    # def __init__(self):
    #     super(LSTMModel, self).__init__()
    #     self.lstm = nn.LSTM(input_size=14, hidden_size=256, batch_first=True)
    #     self.dropout = nn.Dropout(0.2)
    #     self.fc1 = nn.Linear(256, 128)
    #     self.fc2 = nn.Linear(128, 30)
    #
    # def forward(self, x):
    #     x, _ = self.lstm(x)
    #     x = self.dropout(x[:, -1, :])
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return x
    def __init__(self, input_size=14, hidden_size=256, num_stacked_layers=2, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 30)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)
        x = self.fc2(x)
        return x