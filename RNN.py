import torch.nn as nn

default_configs = {'input_size': 2,
                   'hidden_size': 16,
                   'output_size': 1,
                   'num_layers ': 2,
                   'batch_size': 100,
                   'learning_rate': 0.01,
                   'threshold': 1e-3,
                   'num_epochs': 10}

class lstm(nn.Module):
    def __init__(self, input_size = None, hidden_size = None, num_layers = None, output_size = None):
        super(lstm, self).__init__()
        self.input_size = default_configs['input_size'] if input_size is None else input_size
        self.hidden_size = default_configs['hidden_size'] if hidden_size is None else hidden_size
        self.num_layers = default_configs['num_layers '] if num_layers is None else num_layers
        self.output_size = default_configs['output_size'] if output_size is None else output_size

        self.layer1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.reshape(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x