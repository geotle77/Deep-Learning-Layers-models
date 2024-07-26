import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))  # 设置可训练的参数，用于层归一化的缩放
        self.beta = nn.Parameter(torch.zeros(d_model))  # 设置可训练的参数，用于层归一化的平移

        self.eps = eps  # 避免分母为0
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 对最后一个维度求均值，并保持维度。
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps) # 相当于正态分布中的(X-E(X)) / sqrt(Var(X))
        out = self.gamma * out + self.beta #提高模型的性能和灵活性
        return out