import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """
    def __init__(self, vocab_size: int, emb_size: int):
        """
        :param vocab_size: size of vocabulary
        :param emb_size: size of word embedding

        class for token embedding that included positional information
        """
        super(TokenEmbedding, self).__init__(vocab_size, emb_size, padding_idx=0)