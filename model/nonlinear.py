import torch
from torch import nn

class NonLinear(nn.Module):
    def __init__(self, in_nc=3, nc=3200, out_nc=18, sheet_num=4):
        super(NonLinear, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * sheet_num)
        self.sheet_num = sheet_num
        self.out_nc = out_nc

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        output = output.view(-1, self.sheet_num, self.out_nc)
        return output