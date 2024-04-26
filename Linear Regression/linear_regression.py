import torch.nn as nn

sequence_length = 90
prediction_length = 30
batch_size = 128
epochs = 100
device = 'cpu'


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1260, 30)

    def forward(self, x):
        return self.linear(x)
