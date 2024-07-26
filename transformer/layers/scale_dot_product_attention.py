import math 
from torch import nn

class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None,e = 1e-12):
        '''
        Scale Dot Product Attention
        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]
            mask: Mask张量，形状为[B, L_q, L_k]
        Returns:
            output: 经过注意力机制加权后的张量，形状为[B, L_q, D_v]
            attention: 注意力权重
        '''
        batch_size, head,length,d_tensor = k.size()

        # 计算QK^T/sqrt(d_k)
        k_t =k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 计算mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        #使用softmax函数进行归一化处理
        score  = self.softmax(score)

        #计算注意力权重值
        v= score @ v

        return v,score
    