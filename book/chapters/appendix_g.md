# Appendix G — PyTorch Reference

## G.1 Introduction

The snippets below look like a glossary, but several of them hide behavior that causes real bugs if taken at face value: `backward()` *accumulates* rather than assigns, `.to()` is a no-op copy on tensors but an in-place mutation on modules, `.detach()` shares storage with its parent, and `torch.load` executes arbitrary code by default unless told not to. This appendix restates each call with its precise semantics, the aliasing/mutation behavior, and the condition under which the "obvious" reading is wrong.

### G.1.1 Conventions

`x` denotes a `torch.Tensor`, `module` a `torch.nn.Module`, `param` a `torch.nn.Parameter` (itself a `Tensor` subclass with `requires_grad=True` by default), and `loss` a scalar `Tensor` produced by a chain of differentiable operations starting from tensors with `requires_grad=True`. "In-place" below means the operation mutates the receiver's existing storage and returns the same object (or `None`); "returns a new tensor" means a fresh `Tensor` object is allocated, though its storage may still alias the original's (a *view*).

---

## G.2 Tensor Introspection

```python
print(x.shape)      # Tensor shape
print(x.dtype)      # Data type
print(x.device)     # Device
```

- **`x.shape`** returns a `torch.Size`, a subclass of Python `tuple`. It is immutable: `x.shape[0] = 3` raises `TypeError`. It is equivalent to `x.size()`; `x.shape` is the attribute form, `x.size()` the method form (the latter also accepts a dimension argument, `x.size(0)`, for which there is no attribute equivalent).
- **`x.dtype`** is a `torch.dtype` object (e.g. `torch.float32`), not a Python type or string, though it compares equal to nothing else — `x.dtype == torch.float32` works, `x.dtype == 'float32'` does not. The *default* dtype used when constructing a tensor from a Python float (e.g. `torch.tensor([1.0])`) is governed by `torch.get_default_dtype()` / `torch.set_default_dtype(...)`, which affects only tensors constructed *after* the call — pre-existing tensors keep their dtype.
- **`x.device`** is a `torch.device` object with a `.type` (`'cpu'`, `'cuda'`, `'mps'`, ...) and an optional `.index`. A device with `index=None` (e.g. constructed as `torch.device('cuda')`) refers to "the currently active CUDA device" and is resolved at operation time, not at device-construction time — two tensors both reporting `device='cuda'` are not guaranteed to be on the *same* physical device unless their indices match or both were created under the same `torch.cuda.device(...)` context.

---

## G.3 Autograd: `backward()` and `.grad`

```python
loss.backward()     # Compute gradients
print(param.grad)   # Access gradient
```

- **`loss.backward()`** runs reverse-mode automatic differentiation over the dynamically-built computation graph reachable from `loss`, and for every leaf tensor `t` with `requires_grad=True` that graph touches, **adds** $\partial\,\mathrm{loss}/\partial t$ into `t.grad` — it does not assign, it accumulates: `t.grad = t.grad + computed_grad` if `t.grad` already holds a value, or sets `t.grad = computed_grad` if it was `None`. This is intentional (it is what makes gradient accumulation across micro-batches possible) but is the single most common source of silent training bugs: calling `backward()` in a loop without an explicit `optimizer.zero_grad()` (or `param.grad = None`) between iterations sums gradients across iterations instead of replacing them.
- `backward()` requires `loss` to be a scalar (0-dimensional), *or* an explicit `gradient=` argument of the same shape as `loss`, supplying the vector to be used in the vector–Jacobian product $v^\top J$ that reverse-mode AD actually computes — `backward()` on a non-scalar tensor without `gradient=` raises `RuntimeError`.
- By default, the intermediate buffers needed for backpropagation are **freed** once `backward()` completes, so a second `backward()` call over the same graph raises `RuntimeError: Trying to backward through the graph a second time` unless the first call used `retain_graph=True`.
- **`param.grad`** is `None` until some `backward()` call has populated it (not a zero tensor — the distinction matters if you branch on `if param.grad is not None:`). Setting `requires_grad=False` on a parameter after the fact does not clear an existing `.grad`; clearing is a separate, explicit step (`zero_grad()` or `param.grad = None`).

---

## G.4 Hooks

```python
handle = module.register_forward_hook(hook)  # Register hook
```

`register_forward_hook` attaches `hook` to `module` with signature `hook(module, input, output) -> Optional[Tensor]`, invoked every time `module.forward` completes during a forward pass. If `hook` returns a value other than `None`, that value **replaces** `output` for all downstream consumers of `module`'s output within that forward pass — a return value of `None` leaves `output` unmodified. `input` is always a `tuple` of the positional arguments passed to `forward`, even if `forward` takes exactly one argument.

The registration call returns a `RemovableHandle`, not the hook function itself; the hook remains attached to `module` until `handle.remove()` is called explicitly. Forgetting to store and later remove `handle` is a standing memory/behavior leak: the hook keeps firing (and keeps whatever it closes over alive) for the lifetime of `module`, including across unrelated later forward passes. This is distinct from `register_forward_pre_hook` (fires before `forward`, sees only `input`, can modify it before the call) and `register_full_backward_hook` (fires during the backward pass, sees gradients rather than activations) — the three hook types intercept different points in the computation and have different signatures.

---

## G.5 Disabling Gradient Tracking

```python
with torch.no_grad():  # Disable gradients
    output = model(x)
```

`torch.no_grad()` is a context manager that changes how operations *inside* the block are recorded: any tensor produced by an op executed under `no_grad()` has `requires_grad=False` and **no `grad_fn`** — the graph node is never constructed in the first place, rather than being constructed and then discarded. This is different in kind from `.detach()` (§G.6), which severs an *already-existing* tensor from its already-built history but does not prevent history from being built for other tensors, and from setting `requires_grad_(False)` on a leaf, which prevents that specific leaf from accumulating gradients but does not stop the graph from being built for tensors derived from other, still-tracked inputs.

`torch.inference_mode()` is a stricter, generally faster variant of `no_grad()` (skips additional bookkeeping used to support in-place-modification version checks), recommended over `no_grad()` for pure inference where none of the tensors produced inside the block will ever need `.requires_grad_(True)` re-enabled or be used in an autograd-tracked op afterward — tensors created under `inference_mode()` cannot later be made to require grad or used in a graph without an explicit `.clone()`.

---

## G.6 Detaching from the Graph

```python
x.detach()             # Remove from computation graph
```

`x.detach()` returns a **new `Tensor` object** with `requires_grad=False` and no `grad_fn`, but this new tensor **shares the same underlying storage** as `x` — it is a view, not a copy. Consequently:

- Reading `x.detach()` is safe and is the standard way to pull a value out of the graph for logging, metrics, or use as a constant.
- **In-place** modification of `x.detach()` (e.g. `x.detach().add_(1)`, or `x.detach()[0] = 0`) mutates the same storage that `x` uses, which corrupts any autograd computation still pending on `x` — PyTorch's version counter will typically catch this and raise `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` (or an "modified by an inplace operation" error) the next time `backward()` tries to use `x`, but the failure is at `backward()` time, not at the point of the mutation, which makes it easy to introduce far from where the actual bug is.
- If an independent copy is needed, use `x.detach().clone()` (or equivalently `x.clone().detach()` — order does not matter here since neither op depends on the other's output already being detached/copied).

---

## G.7 Determinism

```python
torch.manual_seed(42)  # Set seed
```

`torch.manual_seed(42)` seeds only the **default CPU** random number generator (and, as a convenience, the default CUDA generator *for the current device* — but not other CUDA devices in a multi-GPU setup, which require `torch.cuda.manual_seed_all(42)`). Setting this seed alone is **not** sufficient for bit-for-bit reproducibility, because:

1. Several CUDA kernels (e.g. certain forms of `scatter_add`, some convolution backward algorithms) are inherently nondeterministic regardless of seed, since they rely on unordered floating-point accumulation across threads; `torch.use_deterministic_algorithms(True)` forces PyTorch to raise an error or select a slower deterministic kernel instead of silently proceeding nondeterministically.
2. `torch.backends.cudnn.benchmark = True` (a common performance setting) lets cuDNN auto-tune and select among multiple algorithms based on runtime timing, which can vary run-to-run and change results even at a fixed seed; reproducibility requires `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True`.
3. Seeding `torch`, Python's `random`, and `numpy.random` are three independent RNG states — code that touches any of the latter two (directly, or indirectly via a library such as a `DataLoader`'s default shuffling, which does *not* automatically inherit `torch.manual_seed`) needs its own explicit seed, and multi-worker `DataLoader`s additionally need a `worker_init_fn` to seed each worker process separately, since forking does not guarantee distinct RNG state across workers.

---

## G.8 Saving and Loading

```python
torch.save(obj, path)  # Save tensor/model
torch.load(path)       # Load tensor/model
```

`torch.save` serializes `obj` using Python's `pickle` protocol (with a small amount of additional framing for tensor storages). `torch.load` correspondingly **unpickles** the file, which means it can execute arbitrary Python code embedded in a malicious file — loading a checkpoint from an untrusted source with the historical default behavior is a code-execution risk, not merely a data-parsing one. Recent PyTorch versions default `weights_only=True`, which restricts unpickling to a safe allow-list of tensor/container types and rejects arbitrary object graphs; loading a checkpoint that legitimately contains non-tensor objects (e.g. a full pickled `nn.Module` rather than a `state_dict`) requires explicitly passing `weights_only=False`, at which point the same code-execution caveat applies.

Two distinct things can be saved, with different portability properties:

- `torch.save(model.state_dict(), path)` saves only the parameter/buffer tensors as an `OrderedDict` keyed by name. This is the recommended form: it is robust to refactoring the `nn.Module`'s class definition (as long as parameter names still line up), and loading it back requires reconstructing the model class first (`model.load_state_dict(torch.load(path))`).
- `torch.save(model, path)` pickles the entire `Module` object, including a reference to its class. Loading this back (`torch.load(path)`) requires the *exact* class definition to be importable from the *same module path* at load time — renaming or moving the class, or loading in an environment without the original source file, breaks it.

`torch.load(path, map_location=...)` is the mechanism for loading a checkpoint saved from a CUDA tensor onto a machine without that GPU, or onto a different GPU index — omitting `map_location` when loading a GPU-saved checkpoint on a CPU-only machine raises `RuntimeError` rather than silently falling back to CPU.

---

## G.9 Moving Between Devices

```python
x.to('cuda')           # Move to GPU
x.cpu()                # Move to CPU
```

For a **tensor**, `x.to(device)` returns a **new tensor** if a copy is actually required (different device, or a `dtype=` argument that changes the element type); if `device` (and any requested `dtype`) already match `x`'s current device and dtype, `x.to(...)` is documented to return `x` itself (no copy), so code should not rely on `x.to(device) is not x` as a way to detect whether a move happened. Either way, the original `x` binding is **unchanged** — `x.to('cuda')` on its own line, with the result discarded, does nothing observable; the idiom is always `x = x.to('cuda')`.

`x.cpu()` is exactly `x.to('cpu')` (in fact implemented as a thin wrapper), and is subject to the same same-device-is-a-no-op behavior.

**This is the opposite of `nn.Module.to()`**: `module.to(device)` moves the module's parameters and buffers **in place** (it mutates each `Parameter`/buffer tensor's storage location and returns `module` itself, `self`) — `module.to('cuda')` alone (without reassignment) *does* move the module, and writing `module = module.to('cuda')` is idiomatic but not required for the mutation to take effect, only for readability/chaining. Conflating the two — expecting `x.to('cuda')` to mutate `x` in place because `module.to('cuda')` mutates `module` in place — is a common source of "my tensor is still on CPU" bugs.

---

## G.10 Reproducibility Checklist

For a serious experiment, "record the seed" is necessary but not sufficient. A reproducibility record should capture:

1. **All RNG seeds**, not just `torch.manual_seed`: also `random.seed`, `numpy.random.seed`, `torch.cuda.manual_seed_all` for multi-GPU, and a `DataLoader` `worker_init_fn` if `num_workers > 0` (§G.7).
2. **Determinism flags**: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and — for full CUDA determinism on operations like certain matrix multiplications — the `CUBLAS_WORKSPACE_CONFIG` environment variable, which must be set *before* the CUDA context is initialized (setting it mid-run has no effect).
3. **Software versions**: `torch.__version__`, `torch.version.cuda`, and the cuDNN version (`torch.backends.cudnn.version()`), since numerical results can shift across library versions even with identical seeds and flags — a seed is only reproducible relative to a fixed software stack, not in an absolute sense.
4. **Hardware class**, when bitwise reproducibility is required: floating-point reduction order (and therefore the exact result) can differ across GPU architectures even with identical seeds, flags, and library versions, because kernel scheduling is hardware-dependent.

Recording (1) alone reproduces the *sequence of random numbers drawn*; it does not by itself reproduce the *numerical results* of the computation unless (2)–(4) are also fixed, since nondeterministic kernel selection and floating-point accumulation order are independent of the RNG state.
