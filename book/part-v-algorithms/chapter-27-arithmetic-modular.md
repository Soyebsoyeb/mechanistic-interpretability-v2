# Chapter 27 — Arithmetic and Modular Computation

## Motivation
Synthetic arithmetic has known ground-truth algorithms, ideal for mechanistic study.

## Setup
Train transformers on $(a + b) \mod p$.

## Analysis
- Embeddings: periodic structure
- Attention: position patterns
- MLPs: Fourier-like representations
- Output decoding

## Implementation

```python
def generate_modular_data(p=113, n_samples=10000):
    a = torch.randint(0, p, (n_samples,))
    b = torch.randint(0, p, (n_samples,))
    c = (a + b) % p
    return torch.stack([a, b], dim=1), c

def analyze_periodic_structure(embedding_matrix, p):
    fft = torch.fft.fft(embedding_matrix, dim=0)
    magnitudes = torch.abs(fft)
    return torch.topk(magnitudes.mean(dim=1), 5)
```

## Falsification
Falsified if model does not use Fourier-like representations when predicted.

## Exercises
- **Mathematical**: Prove modular addition via trigonometric identities.
- **Implementation**: Implement Fourier analysis of embeddings.
- **Experimental**: Train and reverse-engineer modular addition transformer.
- **Research**: Do larger $p$ require different circuit structures?

## References
- Nanda, N., et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability."
