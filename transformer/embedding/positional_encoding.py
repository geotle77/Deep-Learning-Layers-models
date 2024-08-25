import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len,device):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        self.encoding =  torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False # set encoding as non-trainable

        position = torch.arange(0, max_len).to(device)
        position = position.unsqueeze(dim=1)

        #1D => 2D unsqueeze to represent the position

        _2i = torch.arange(0, d_model, 2).to(device).float()
        #"i" is index of d_model(e.g. embedding size = 50, 'i' = [0,50])
        #"step = 2" means "i" mutiplies 2 times

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))
        #sin to even index, cos to odd index

    def forward(self, x):
        batch_size,seq_len = x.size()

        return self.encoding[:seq_len, :]