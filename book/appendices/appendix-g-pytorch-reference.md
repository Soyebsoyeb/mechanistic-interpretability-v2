# Appendix G — PyTorch Reference

```python
print(x.shape)      # Tensor shape
print(x.dtype)      # Data type
print(x.device)     # Device

loss.backward()     # Compute gradients
print(param.grad)   # Access gradient

handle = module.register_forward_hook(hook)  # Register hook

with torch.no_grad():  # Disable gradients
    output = model(x)

torch.manual_seed(42)  # Set seed

torch.save(obj, path)  # Save tensor/model
torch.load(path)       # Load tensor/model

x.to('cuda')           # Move to GPU
x.cpu()                # Move to CPU
x.detach()             # Remove from computation graph
```

## Reproducibility
For serious experiments, record all random seeds and software versions.
