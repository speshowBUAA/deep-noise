import torch
from torch import nn

class NonLinear(nn.Module):
    def __init__(self, in_nc=3, nc=100, out_nc=1):
        super(NonLinear, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)


    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)

        return output