import torch
import pytest
from src.models.transformer import attention, TransformerBlock

def test_attention_dimensions():
    batch, seq, d_model, d_head = 2, 10, 64, 16
    x = torch.randn(batch, seq, d_model)
    W_q = torch.randn(d_model, d_head)
    W_k = torch.randn(d_model, d_head)
    W_v = torch.randn(d_model, d_head)
    W_o = torch.randn(d_head, d_model)
    output, weights = attention(x, W_q, W_k, W_v, W_o)
    assert output.shape == (batch, seq, d_model)
    assert weights.shape == (batch, seq, seq)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq), atol=1e-6)

def test_transformer_block():
    batch, seq, d_model = 2, 10, 64
    x = torch.randn(batch, seq, d_model)
    block = TransformerBlock(d_model, num_heads=8, d_mlp=256)
    output = block(x)
    assert output.shape == (batch, seq, d_model)
    assert not torch.isnan(output).any()
