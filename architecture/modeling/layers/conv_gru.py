import torch
import torch.nn as nn

class ConvGRU(nn.Module):
    def __init__(self,
                 in_planes: int,
                 hidden_planes: int):
        super(ConvGRU, self).__init__()

        self.in_planes = in_planes
        self.hidden_planes = hidden_planes

        fuse_planes = in_planes + hidden_planes

        self.convz = nn.Conv2d(fuse_planes, hidden_planes, 3, padding=1)
        self.convr = nn.Conv2d(fuse_planes, hidden_planes, 3, padding=1)
        self.convq = nn.Conv2d(fuse_planes, hidden_planes, 3, padding=1)

    def forward(self, last_hidden, x):
        hx = torch.cat((last_hidden, x), dim=1)

        update_gate = torch.sigmoid(self.convz(hx))
        reset_gate  = torch.sigmoid(self.convr(hx))
        cur_hidden  = torch.tanh(self.convq(torch.cat((reset_gate * last_hidden, x), dim=1)))

        hidden = (1 - update_gate) * last_hidden + update_gate * cur_hidden

        return hidden
