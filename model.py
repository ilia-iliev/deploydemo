import torch
from preprocessing import Tokenizer
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        # tensor to hold positional encoding
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)) * (-math.log(10000.)/d_model)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('positional_encoding', pe)
    

    def forward(self, x):
        return x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)


class InputEmbeddings(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int, padding_idx=-1) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim=d_model, padding_idx=padding_idx)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class SentimentClassifier(torch.nn.Module):
    """
    Implements a sentiment classifier using a transformer-based architecture.

    Args:
        d_model (int): The dimensionality of the input and output embeddings.
        n_heads (int): The number of attention heads in the transformer encoder layers.
        vocab_size (int): The size of the vocabulary.
        seq_len (int): The maximum sequence length.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, 2), representing the predicted sentiment scores"""
    def __init__(self, d_model, n_heads, vocab_size, seq_len) -> None:
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.context_size = seq_len
        self.embeddings = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        self.classifier = torch.nn.Linear(d_model, 2)
    
    def forward(self, x, mask=None):
        x = self.embeddings(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class SentimentInference():
    def __init__(self, model: SentimentClassifier, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inp: str):
        tokens = self.tokenizer.tokenize(inp)[:self.model.context_size+1]
        res = self.model(torch.IntTensor([tokens]))
        return res.argmax().item()
    
    @staticmethod
    def load():
        m = torch.load("model.pt")
        t = Tokenizer.load("tokenizer.pkl")
        return SentimentInference(m, t)
