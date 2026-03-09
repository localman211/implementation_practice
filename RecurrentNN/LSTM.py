import torch
import torch.nn as nn
import torch.nn.functional as F

#many_to_one 구현
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear = nn.Linear(in_features = input_size+hidden_size, out_features = hidden_size*4)
        self.out = nn.Linear(in_features = hidden_size, out_features = output_size)
        self.hidden_size = hidden_size


    def forward(self, x):
        hidden_state = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        cell_state = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        h = self.hidden_size

        for t in range(x.size(1)):
            seq = torch.cat((x[:,t], hidden_state), dim=1)
            f = self.linear(seq)
            cell_state = (cell_state*F.sigmoid(f[:,:h]) + F.sigmoid(f[:,h:h*2])*F.tanh(f[:,h*2:h*3]))
            hidden_state = (F.sigmoid(f[:,h*3:])*F.tanh(cell_state))

        outputs = self.out(hidden_state)

        return outputs