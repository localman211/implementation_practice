import torch
import torch.nn as nn
import torch.nn.functional as F

#many to one 구현
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.x_to_r = nn.Linear(in_features = input_size, out_features = hidden_size)
        self.h_to_r = nn.Linear(in_features = hidden_size, out_features = hidden_size)

        self.x_to_c = nn.Linear(in_features = input_size, out_features = hidden_size)
        self.h_to_c = nn.Linear(in_features = hidden_size, out_features = hidden_size)

        self.x_to_z = nn.Linear(in_features = input_size, out_features = hidden_size)
        self.h_to_z = nn.Linear(in_features = hidden_size, out_features = hidden_size)

        self.out = nn.Linear(in_features = hidden_size, out_features = output_size)

        self.hidden_size = hidden_size

    def forward(self, x):
        h = self.hidden_size
        hidden_state = torch.zeros(x.size(0), h, device=x.device, dtype=x.dtype)

        for t in range(x.size(1)):
            reset_gate = F.sigmoid(self.x_to_r(x[:,t]) + self.h_to_r(hidden_state))
            candidate = F.tanh(self.x_to_c(x[:,t]) + reset_gate*self.h_to_c(hidden_state))
            update_gate = F.sigmoid(self.x_to_z(x[:,t]) + self.h_to_z(hidden_state))

            hidden_state = (1-update_gate)*hidden_state + update_gate*candidate

        output = self.out(hidden_state)
        return output