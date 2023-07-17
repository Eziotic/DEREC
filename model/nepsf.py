import torch 
from torch import nn
from torch.nn import functional as F

class D2DNet(nn.Module):
    def __init__(self, input_ch=2, output_ch=1, D=2, W=16):
        super(D2DNet, self).__init__()
        self.enc = nn.ModuleList(
            [nn.Conv2d(input_ch, W, kernel_size=1)] + [nn.Conv2d(W, W, kernel_size=1) for i in range(D-1)])

        self.dec = nn.Conv2d(W, output_ch, kernel_size=1)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.enc):
            h = self.enc[i](h)
            h = F.relu(h)
        output = self.dec(h)
        return output
