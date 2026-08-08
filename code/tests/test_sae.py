import torch
import pytest
from src.sae.model import SparseAutoencoder

def test_sae_forward():
    d_in, d_hidden = 64, 512
    sae = SparseAutoencoder(d_in, d_hidden)
    x = torch.randn(32, d_in)
    out = sae(x)
    assert out['x_hat'].shape == (32, d_in)
    assert out['z'].shape == (32, d_hidden)
    assert out['total_loss'].item() >= 0
    assert 0 <= out['mean_l0'] <= 1

def test_sae_reconstruction():
    d_in, d_hidden = 64, 512
    sae = SparseAutoencoder(d_in, d_hidden, sparsity_lambda=0.0)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    x = torch.eye(d_in)
    for _ in range(500):
        opt.zero_grad()
        out = sae(x)
        out['total_loss'].backward()
        opt.step()
        sae.normalize_decoder()
    out = sae(x)
    recon_error = (out['x_hat'] - x).abs().mean()
    assert recon_error < 0.1
