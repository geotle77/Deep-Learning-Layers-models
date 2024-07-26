from torch import nn

class PositionwiseFeedforward(nn.Module):
    '''
     位置前馈网络通过引入非线性变换和独立处理每个位置，增强了 Transformer 模型的表达能力和特征提取能力。
     对每个位置的token的特征进行非线性变换，然后再进行线性变换。
    '''
    def __init__(self,d_model,hidden,drop_prob=0.1):
        super(PositionwiseFeedforward,self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x