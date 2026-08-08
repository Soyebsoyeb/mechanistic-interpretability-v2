import torch

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
