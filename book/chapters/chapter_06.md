# Chapter 6 — Transformer Architecture from First Principles

## Motivation

Transformers are the dominant architecture in modern deep learning. To reverse engineer them, we must understand their mathematical structure with complete precision. Every tensor dimension, every matrix multiplication, and every nonlinear transformation must be accounted for. This chapter derives the transformer forward pass from first principles, with no steps omitted.

## Learning Objectives

- Derive the transformer forward pass from first principles with complete dimensional accounting
- Implement a transformer attention mechanism from scratch with full documentation
- Verify implementation against a reference implementation
- Understand the role of each component in the computational graph
- Compute FLOP counts and memory requirements

## Formal Definition: Single Attention Head

Let $X \in \mathbb{R}^{n \times d}$ be the input sequence of $n$ tokens in $d$ dimensions. Each row $x_i \in \mathbb{R}^d$ is the representation of token $i$.

For one attention head with head dimension $d_h$:

### Query, Key, Value Projections

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$.

**Dimensional accounting:**
- $X$: $[n \times d]$
- $W_Q, W_K, W_V$: $[d \times d_h]$
- $Q, K, V$: $[n \times d_h]$

### Attention Scores

$$S = \frac{QK^\top}{\sqrt{d_h}}$$

where $S \in \mathbb{R}^{n \times n}$.

The scaling factor $\sqrt{d_h}$ prevents dot products from growing too large in magnitude, which would push the softmax into extremely saturated regions where gradients vanish. To see why, note that if $q_i, k_j \sim \mathcal{N}(0, I_{d_h})$, then $\mathbb{E}[q_i^\top k_j] = 0$ and $\text{Var}(q_i^\top k_j) = d_h$. Dividing by $\sqrt{d_h}$ normalizes the variance to 1.

### Attention Weights

$$A = \text{softmax}(S)$$

where $A \in \mathbb{R}^{n \times n}$ and each row sums to 1:

$$\sum_{j=1}^n A_{ij} = 1 \quad \text{for all } i$$

The softmax is applied row-wise:

$$A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^n \exp(S_{ik})}$$

### Head Output

$$Z = AV$$

where $Z \in \mathbb{R}^{n \times d_h}$. Each row $z_i = \sum_{j=1}^n A_{ij} v_j$ is a weighted average of value vectors, weighted by attention scores.

### Output Projection

$$O = Z W_O$$

where $W_O \in \mathbb{R}^{d_h \times d}$ and $O \in \mathbb{R}^{n \times d}$.

### Multi-Head Attention

With $h$ heads, we have $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_h}$ for $i = 1, \ldots, h$, where typically $d_h = d/h$.

$$\text{MultiHead}(X) = \text{Concat}(Z^{(1)}, \ldots, Z^{(h)}) W_O^{\text{multi}}$$

where $W_O^{\text{multi}} \in \mathbb{R}^{d \times d}$.

**Total parameters in attention**: $3hd \cdot d_h + d^2 = 4d^2$ when $d_h = d/h$.

## Implementation From Scratch

```python
import torch
import torch.nn as nn
import math

def attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple:
    """Single-head attention from scratch with full dimension documentation.

    Args:
        x: [batch, seq_len, d_model]
        W_q: [d_model, d_head]
        W_k: [d_model, d_head]
        W_v: [d_model, d_head]
        W_o: [d_head, d_model]
        mask: [seq_len, seq_len], optional causal or padding mask

    Returns:
        output: [batch, seq_len, d_model]
        attention_weights: [batch, seq_len, seq_len]
    """
    batch, seq_len, d_model = x.shape
    d_head = W_q.shape[1]

    # Project to Q, K, V
    Q = x @ W_q  # [batch, seq_len, d_head]
    K = x @ W_k  # [batch, seq_len, d_head]
    V = x @ W_v  # [batch, seq_len, d_head]

    # Compute scores: Q @ K^T
    scores = Q @ K.transpose(-1, -2)  # [batch, seq_len, seq_len]
    scores = scores / math.sqrt(d_head)

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over keys (dim=-1)
    weights = torch.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]

    # Apply attention to values
    z = weights @ V  # [batch, seq_len, d_head]

    # Output projection
    output = z @ W_o  # [batch, seq_len, d_model]

    return output, weights


class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and MLP.

    Architecture:
        x -> LayerNorm -> MultiHeadAttn -> Residual -> 
        LayerNorm -> MLP -> Residual -> output
    """

    def __init__(self, d_model: int, num_heads: int, d_mlp: int, 
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Attention weights: combined Q, K, V for all heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # MLP weights
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

        # Layer norms (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [seq_len, seq_len], optional

        Returns:
            [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Attention sublayer
        attn_input = self.ln1(x)
        qkv = self.W_qkv(attn_input)  # [batch, seq, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [batch, seq, d_model]

        # Reshape for multi-head: [batch, seq, heads, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Attention scores: [batch, heads, seq, seq]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ v  # [batch, heads, seq, head_dim]

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.W_o(attn_out)

        # Residual connection 1
        x = x + self.dropout(attn_out)

        # MLP sublayer
        mlp_input = self.ln2(x)
        mlp_out = self.W_out(torch.relu(self.W_in(mlp_input)))

        # Residual connection 2
        x = x + self.dropout(mlp_out)

        return x
```

## Verification

```python
def test_attention_dimensions():
    """Verify all tensor dimensions in attention."""
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
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print("All dimension tests passed.")


def test_causal_masking():
    """Verify causal mask prevents attending to future tokens."""
    batch, seq, d_model, d_head = 1, 5, 64, 16
    x = torch.randn(batch, seq, d_model)
    W_q = torch.randn(d_model, d_head)
    W_k = torch.randn(d_model, d_head)
    W_v = torch.randn(d_model, d_head)
    W_o = torch.randn(d_head, d_model)

    causal_mask = torch.tril(torch.ones(seq, seq))
    output, weights = attention(x, W_q, W_k, W_v, W_o, causal_mask)

    # Upper triangle should be zero (or very small due to softmax)
    upper = torch.triu(torch.ones(seq, seq), diagonal=1)
    assert (weights[0] * upper).sum() < 1e-6
    print("Causal masking test passed.")
```

## Measurement: Attention Pattern Analysis

Given attention weights $A \in \mathbb{R}^{n \times n}$:

- **Entropy**: $H(A_i) = -\sum_j A_{ij} \log A_{ij}$. Low entropy indicates focused attention; high entropy indicates diffuse attention.
- **Diagonal mass**: $\frac{1}{n}\sum_i A_{ii}$. High diagonal mass suggests position-specific processing.
- **Copying score**: For induction heads, measure $A_{j,i}$ where $x_j = x_i$ and $j > i$.
- **Positional bias**: $\frac{1}{n-1}\sum_{i=2}^n A_{i, i-1}$ (attention to previous token).

## Intervention: Head Ablation

```python
def ablate_head(model, layer_idx, head_idx, inputs):
    """Zero out a specific attention head.

    Args:
        model: Transformer model
        layer_idx: int, layer index
        head_idx: int, head index within layer
        inputs: torch.Tensor

    Returns:
        Model output with head ablated
    """
    def hook_fn(module, input, output):
        # output is attention output [batch, seq, d_model]
        batch, seq, d_model = output.shape
        d_head = d_model // module.num_heads

        # Reshape to separate heads
        out_heads = output.view(batch, seq, module.num_heads, d_head)
        out_heads[:, :, head_idx, :] = 0

        return out_heads.view(batch, seq, d_model)

    target_layer = model.blocks[layer_idx].attn
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        result = model(inputs)

    handle.remove()
    return result
```

## Falsification

A claimed attention mechanism is falsified if:
- The attention pattern does not match the claimed algorithm on controlled inputs
- Ablating the head does not change the claimed behavior
- The QK and OV matrices do not have the structure predicted by the hypothesis
- The attention pattern is not stable across different random seeds or input samples

## Exercises

### Mathematical
1. Prove that $\text{softmax}(S + c) = \text{softmax}(S)$ for any scalar $c$ added to all elements of a row. Why is this property important for numerical stability?
2. Show that multi-head attention with $h$ heads and $d_h = d/h$ has the same number of parameters as single-head attention with dimension $d$ when $W_O^{\text{multi}}$ is included.
3. Compute the FLOP count for a single attention layer with $n$ tokens, $d$ dimensions, and $h$ heads. How does it scale with $n$?
4. Prove that the attention output $Z$ is Lipschitz continuous with respect to $Q$ and $K$ with constant $\leq 2/\sqrt{d_h}$.

### Implementation
5. Implement multi-head attention using einsum operations instead of reshape/transpose. Compare readability and performance.
6. Write a comprehensive test suite verifying that your attention implementation satisfies: (a) row sums to 1, (b) causal masking works, (c) output dimension matches input, (d) gradient flow is correct.
7. Implement a function that computes the exact FLOP count for a transformer forward pass given architecture parameters.

### Experimental
8. Train a 2-layer transformer on a simple task. Visualize attention patterns for each head. Classify heads by their attention entropy and diagonal mass.
9. Measure how attention patterns change during training. Do they converge quickly or gradually?

### Research
10. Investigate whether attention heads in early layers attend primarily to adjacent positions (local) while late-layer heads attend to distant positions (global). Quantify this using attention range statistics.
11. Study whether attention heads specialize during training or are randomly assigned their functions. Design an experiment to test this.

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
