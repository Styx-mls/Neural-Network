# MiniTorch: A Minimal Neural Network Framework from Scratch

Welcome to **MiniTorch** — a fully functional deep learning framework written from scratch in Python using only `numpy`.

This framework features:
- Autograd engine
- Tensor class with backprop
- PyTorch-style layers (`Linear`, `ReLU`, `Dropout`, etc.)
- Loss functions (`MSELoss`, `CrossEntropyLoss`)
- Optimizers (`SGD`, `Adam`)
- Model saving, loading
- Full support for forward/backward chaining and composition via `Sequential`

---

## Features

| Component | Description |
|----------|-------------|
| `Tensor` | Core data structure with support for gradients and operations |
| `Module` | Base class for all layers (just like `nn.Module`) |
| `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Flatten`, `Residual` | Common layers for building networks |
| `MSELoss`, `CrossEntropyLoss` | Standard loss functions |
| `SGD`, `Adam` | Gradient descent optimizers |
| `Sequential` | Model container to chain layers |
| `no_grad` | Context manager for inference mode |
| `save()` / `load()` | Model checkpointing support |

---
