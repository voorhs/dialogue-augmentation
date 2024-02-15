from torch import nn
import torch
import torch.nn.functional as F


class Projector(nn.Module):
    """Fully-Connected 2-layer Linear Model. Taken from linking prediction paper code."""

    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        
        self.linear = nn.Linear(input_size, input_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, output_size, bias=False)

        torch.nn.init.orthogonal_(self.linear.weight)
        torch.nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        
        x = self.linear(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)

        return x
