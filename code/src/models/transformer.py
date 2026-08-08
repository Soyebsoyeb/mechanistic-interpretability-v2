import torch
import torch.nn as nn
import math

def attention(x, W_q, W_k, W_v, W_o, mask=None):
    batch, seq_len, d_model = x.shape
    d_head = W_q.shape[1]
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K.transpose(-1, -2)
    scores = scores / math.sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    z = weights @ V
    output = z @ W_o
    return output, weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_mlp, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_k = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_o = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * 0.02)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = (x @ self.W_q).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = (x @ self.W_k).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = (x @ self.W_v).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ V
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = attn_out @ self.W_o
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        mlp_out = torch.relu(x @ self.W_in + self.b_in)
        mlp_out = mlp_out @ self.W_out + self.b_out
        x = x + self.dropout(mlp_out)
        x = self.ln2(x)
        return x
