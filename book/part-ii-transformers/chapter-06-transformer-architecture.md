# Chapter 6 — Transformer Architecture from First Principles

## Motivation
Transformers are the dominant architecture. To reverse engineer them, we must understand their mathematical structure with complete precision.

## Single Attention Head

Let $X \in \mathbb{R}^{n \times d}$ be the input sequence.

**Projections:**
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$, so $Q, K, V \in \mathbb{R}^{n \times d_h}$.

**Scores:**
$$S = \frac{QK^\top}{\sqrt{d_h}} \in \mathbb{R}^{n \times n}$$

Scaling by $\sqrt{d_h}$ prevents softmax saturation.

**Weights:**
$$A = \text{softmax}(S) \in \mathbb{R}^{n \times n}$$

**Output:**
$$Z = AV \in \mathbb{R}^{n \times d_h}$$
$$O = Z W_O \in \mathbb{R}^{n \times d}$$

## Multi-Head Attention

With $h$ heads, $d_h = d/h$:
$$\text{MultiHead}(X) = \text{Concat}(Z^{(1)}, \ldots, Z^{(h)}) W_O^{\text{multi}}$$

## Implementation

```python
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
```

## Verification
```python
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
```

## Measurement
- Entropy: $H(A_i) = -\sum_j A_{ij} \log A_{ij}$
- Diagonal mass: $\sum_i A_{ii}$
- Copying score: $A_{j,i}$ where $x_j = x_i$ and $j > i$

## Intervention
```python
def ablate_head(model, layer_idx, head_idx, inputs):
    def hook_fn(module, input, output):
        batch, seq, d_model = output.shape
        d_head = d_model // module.num_heads
        out_heads = output.view(batch, seq, module.num_heads, d_head)
        out_heads[:, :, head_idx, :] = 0
        return out_heads.view(batch, seq, d_model)
    target_layer = model.blocks[layer_idx].attn
    handle = target_layer.register_forward_hook(hook_fn)
    result = model(inputs)
    handle.remove()
    return result
```

## Falsification
Falsified if attention pattern does not match claimed algorithm, or if ablation does not change behavior.

## Exercises
- **Mathematical**: Prove softmax shift invariance. Show multi-head has same parameters as single-head.
- **Implementation**: Implement multi-head with einsum. Write comprehensive test suite.
- **Experimental**: Train 2-layer transformer. Visualize attention patterns. Classify by entropy.
- **Research**: Do early-layer heads attend locally, late-layer heads globally?

## References
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
