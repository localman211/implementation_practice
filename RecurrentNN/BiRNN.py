import torch
import torch.nn as nn
import torch.nn.functional as F

class BiRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.brnn = BiRNN(embedding_size, hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        return self.brnn(x)

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.x_to_s = nn.Linear(input_size, hidden_size)
        self.s_to_end = nn.Linear(hidden_size, hidden_size)
        self.end_to_s = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size*2, output_size)

        self.hidden_size = hidden_size

    def forward(self, x):
        hidden_start_to_end = torch.randn(x.size(2), self.hidden_size, device=x.device, dtype=x.dtype)
        hidden_end_to_start = torch.randn(x.size(2), self.hidden_size, device=x.device, dtype=x.dtype)

        start_output = []
        #순방향
        for t in range(x.size(1)):
            hidden_start_to_end = F.sigmoid(self.x_to_s(x[:,t]) + self.s_to_end(hidden_start_to_end))
            start_output.append(hidden_start_to_end)

        start_output = torch.stack(start_output, dim=1)

        end_output = []
        #역방향
        for t in range(x.size(1)-1, -1, -1):
            hidden_end_to_start = F.sigmoid(self.x_to_s(x[:,t]) + self.s_to_end(hidden_end_to_start))
            end_output.append(hidden_end_to_start)

        end_output = torch.stack(end_output, dim=1)
        end_output = torch.flip(end_output, dims=[1])

        #pytorch의 CrossEntropy에서 softmax 기능이 있기에 model에서는 구현되지 않는다.
        outputs = self.out(end_output + start_output)
        return outputs