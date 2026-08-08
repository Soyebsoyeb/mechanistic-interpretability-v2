import torch
import pytest
from src.hooks.manager import HookManager
from src.models.transformer import TransformerBlock

def test_hook_manager():
    block = TransformerBlock(64, 8, 256)
    x = torch.randn(2, 10, 64)
    manager = HookManager()
    manager.add_hook(block, 'ln1')
    _ = block(x)
    assert 'ln1' in manager.cache
    assert manager.cache['ln1'].shape == (2, 10, 64)
    manager.remove_all()
    assert len(manager.hooks) == 0
