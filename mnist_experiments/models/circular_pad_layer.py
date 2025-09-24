# code from https://github.com/achaman2/truly_shift_invariant_cnns/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F


class circular_pad(nn.Module):
    def __init__(self, padding = (1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding
        
    def forward(self, x):
            
        return F.pad(x, pad = self.pad_sizes , mode = 'circular')