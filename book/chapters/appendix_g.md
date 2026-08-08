# Appendix G — PyTorch Reference

## Tensor Inspection
```python
print(x.shape)      # Tensor shape
print(x.dtype)      # Data type
print(x.device)     # Device
print(x.stride())   # Memory layout
```

## Gradients
```python
loss.backward()     # Compute gradients
print(param.grad)   # Access gradient
param.grad.zero_()  # Clear gradient
```

## Hooks
```python
handle = module.register_forward_hook(hook)
handle = module.register_full_backward_hook(hook)
handle.remove()
```

## Context Managers
```python
with torch.no_grad():       # Disable gradients
with torch.enable_grad():   # Enable gradients
with torch.cuda.amp.autocast():  # Mixed precision
```

## Reproducibility
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Device Management
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
model = model.to(device)
torch.cuda.empty_cache()
```

## Saving and Loading
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```
