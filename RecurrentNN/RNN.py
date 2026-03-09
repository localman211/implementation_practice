import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.x_to_s = nn.Linear(in_features = input_size, out_features = hidden_size)
        self.s_to_s = nn.Linear(in_features = hidden_size, out_features = hidden_size)
        self.out = nn.Linear(in_features = hidden_size, out_features = output_size)
        self.hidden_size = hidden_size

    #many to many가 아닌 many to one을 구현했기 때문에 배열을 사용하지 않음. many to many 구현 시 cat 대신 stack 함수 사용이 효율적
    def forward(self, x):
        #논문에서는 0.1과 같은 작은 수로 하라고 했음. 현대에는 Adam 옵티마이저, Xavier/Kaiming 가중치 초기화로 0 값을 넣어도 됨
        hidden_state = torch.zeros((x.size(0), self.hidden_size), device=x.device, dtype=x.dtype)

        for t in range(x.size(1)):
            hidden_state = F.tanh(self.x_to_s(x[:,t]) + self.s_to_s(hidden_state))

        output = self.out(hidden_state)
        return output