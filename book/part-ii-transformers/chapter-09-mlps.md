# Chapter 9 — MLPs and Feature Transformation

## Motivation
MLP layers transform representations through nonlinear feature expansion.

## Formal Definition
$$m(x) = W_{\text{out}} \sigma(W_{\text{in}} x + b_{\text{in}}) + b_{\text{out}}$$

Let $h = \sigma(W_{\text{in}} x + b_{\text{in}})$. Then:
$$W_{\text{out}} h = \sum_{i=1}^{d_{\text{mlp}}} h_i w_i^{\text{out}}$$

Each $h_i$ gates an output direction $w_i^{\text{out}}$.

## Implementation

```python
def mlp_forward(x, W_in, b_in, W_out, b_out, activation=torch.relu):
    h = activation(x @ W_in + b_in)
    return h @ W_out + b_out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, activation=torch.relu):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * 0.02)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.activation = activation

    def forward(self, x):
        h = self.activation(x @ self.W_in + self.b_in)
        return h @ self.W_out + self.b_out

def analyze_mlp_neurons(W_in, W_out, top_k=20):
    in_norms = W_in.norm(dim=0)
    out_norms = W_out.norm(dim=1)
    strength = in_norms * out_norms
    return torch.topk(strength, top_k)
```

## Measurement
- Neuron activation distribution
- Feature selectivity
- Effective dimension

## Intervention
```python
def ablate_mlp_neuron(model, layer_idx, neuron_idx, inputs):
    def hook_fn(module, input, output):
        output[:, :, neuron_idx] = 0
        return output
    handle = model.blocks[layer_idx].mlp.register_forward_hook(hook_fn)
    result = model(inputs)
    handle.remove()
    return result
```

## Falsification
Falsified if ablation does not change behavior, or if max-activating examples mismatch claimed feature.

## Exercises
- **Mathematical**: Prove MLP with ReLU is piecewise linear.
- **Implementation**: Compute feature circuit through MLP.
- **Experimental**: Identify bigram-activated neurons, ablate, measure change.
- **Research**: Compare early vs. late layer MLP sparsity.

## References
- Geva, M., et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories."
