from model import SentimentClassifier
import torch

def test_model_is_runnable():
    vocab_size=300
    batch = 4
    seq_len = 20
    inp = torch.randint(0, vocab_size, (batch, seq_len))
    model = SentimentClassifier(d_model=12, n_heads=4, vocab_size=vocab_size, seq_len=seq_len)

    res = model(inp)
    print(res)
    assert res.shape == (4, 3)
