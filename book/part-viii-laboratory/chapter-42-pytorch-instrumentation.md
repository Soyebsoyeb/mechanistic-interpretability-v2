# Chapter 42 — PyTorch Instrumentation

## Motivation
Reliable interpretability requires robust instrumentation for extracting and manipulating internal model states without side effects.

## Basic Hook
```python
cache = {}
def save_activation(name):
    def hook(module, inputs, output):
        cache[name] = output.detach().cpu()
    return hook

handle = module.register_forward_hook(save_activation("layer_3"))
# ... run model ...
handle.remove()
```

## Production Requirements
- Memory management (detach, cpu, delete)
- Device handling
- Hook lifecycle (always remove)
- No memory leaks

## Implementation
```python
class HookManager:
    def __init__(self):
        self.cache = {}
        self.hooks = []

    def add_hook(self, model, path, hook_type='forward'):
        module = self._get_module(model, path)
        if hook_type == 'forward':
            handle = module.register_forward_hook(self._make_hook(path))
        else:
            handle = module.register_full_backward_hook(self._make_hook(path))
        self.hooks.append(handle)

    def _get_module(self, model, path):
        parts = path.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                self.cache[name] = output[0].detach().cpu()
            else:
                self.cache[name] = output.detach().cpu()
        return hook

    def remove_all(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_cache(self):
        self.cache.clear()
        torch.cuda.empty_cache()
```

## Exercises
- **Implementation**: Build hook manager supporting nested modules.
- **Experimental**: Profile memory usage with and without hooks.

## References
- PyTorch Documentation: torch.nn.Module.register_forward_hook
