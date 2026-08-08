import torch

class ActivationCache:
    def __init__(self):
        self.activations = {}
        self.metadata = {}

    def store(self, key, tensor, meta=None):
        self.activations[key] = tensor.detach().cpu()
        self.metadata[key] = meta or {}

    def get(self, key, device='cpu'):
        return self.activations[key].to(device)

    def save(self, path):
        torch.save({"activations": self.activations, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        cache = cls()
        cache.activations = data["activations"]
        cache.metadata = data["metadata"]
        return cache
