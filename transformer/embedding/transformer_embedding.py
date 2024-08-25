import torch.nn as nn

from transformer.embedding.positional_encoding import PositionalEncoding
from transformer.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model,max_len,drop_prob, device):
        """
        :param vocab_size: size of vocabulary
        :param emb_size: size of word embedding
        :param max_len: max length of input sentence
        :param device: device - 'cuda' or 'cpu'
        """
        super(TransformerEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        :param x: input sentence
        :return: embedded sentence
        """
        tok_embedding = self.token(x)
        pos_embedding = self.position(x)
        return self.drop_out(tok_embedding + pos_embedding)