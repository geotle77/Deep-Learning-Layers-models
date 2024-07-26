from torch import nn

from transformer.layers.multi_head_attention import ScaleDotProductAttention



class MultiHeadAttention(nn.module):

    def __init__(self,d_model,num_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_concat = nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        # 乘以权重矩阵
        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        # 拆分为多头
        q,k,v = self.split(q),self.split(k),self.split(v)
        # 通过缩放点积注意力机制
        output,attention = self.attention(q,k,v,mask)
        # 多头拼接
        output = self.concat(output)
        output= self.w_concat(output)

        # 可视化注意力权重
        #TODO
        return output


    def split(self,tensor):
        '''
        按照头数进行拆分
        '''
        batch_size,length,d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)#对张量进行重塑并转置
        return tensor
    
    def concat(self,tensor):
        '''
        拼接多头
        '''
        batch_size,n_head,length,d_tensor = tensor.size()
        d_model = n_head * d_tensor

        tensor = tensor.transpose(1,2).contiguous().view(batch_size,length,d_model)#contiguous 方法确保张量在内存中是连续存储的，这样 view 方法才能正确地重塑张量的形状。
        return tensor