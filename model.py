"""
Model definitions for the metaatom GUI app.

This file intentionally keeps class names and structure compatible with existing
PyTorch Lightning checkpoints (e.g. `tandemnet.ckpt`, `forward_real.ckpt`, etc.).

The networks predict:
- phase-related outputs (via phaseMLP; final Tanh, clamped to [-1, 1])
- reflectivity (via refMLP; final Sigmoid, clamped to [0, 1])
- inverse design parameters (via InverseMLP; clamped to [0, 1])
"""

import torch
from torch import nn
from pytorch_lightning import LightningModule


class phaseMLP(LightningModule):
    def __init__(self, input_dim=4, output_dim=2, lr=0.0001, k=16, l=7):
        super().__init__()

        modules = []
        modules.append(nn.Linear(input_dim, k * 2**l))
        modules.append(nn.LeakyReLU(0.3))

        for i in range(l):
            modules.append(nn.Linear(k * 2 ** (l - i), k * 2 ** (l - i - 1)))
            modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Linear(k, output_dim))
        modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)
        self.lr = lr

    def forward(self, x):
        output = self.layers(x)
        output = torch.clamp(output, min=-1, max=1)
        return output


class refMLP(LightningModule):
    def __init__(self, input_dim=4, output_dim=2, lr=0.0001, k=16, l=7):
        super().__init__()

        modules = []
        modules.append(nn.Linear(input_dim, k * 2**l))
        modules.append(nn.LeakyReLU(0.3))

        for i in range(l):
            modules.append(nn.Linear(k * 2 ** (l - i), k * 2 ** (l - i - 1)))
            modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Linear(k, output_dim))
        modules.append(nn.Sigmoid())

        self.layers = nn.Sequential(*modules)
        self.lr = lr

    def forward(self, x):
        output = self.layers(x)
        output = torch.clamp(output, min=0, max=1)
        return output


class InverseMLP(LightningModule):
    def __init__(self, input_dim=2, output_dim=4, lr=0.0001, k=16, l=7):
        super().__init__()

        modules = []
        modules.append(nn.Linear(input_dim, k))
        modules.append(nn.LeakyReLU(0.3))

        for i in range(l):
            modules.append(nn.Linear(k * 2**i, k * 2 ** (i + 1)))
            modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Linear(k * 2**l, output_dim))

        self.layers = nn.Sequential(*modules)
        self.lr = lr

    def forward(self, x):
        output = self.layers(x)
        output = torch.clamp(output, min=0, max=1)
        return output


class Tandem(LightningModule):
    """
    Tandem network used in the GUI for inverse design.

    Important: keeps checkpoint-loading behavior from the original implementation
    so that `Tandem.load_from_checkpoint('tandemnet.ckpt')` continues to work.
    """

    def __init__(self):
        super().__init__()

        # These three lines are intentionally kept as-is for compatibility. [file:9]
        self.forward_Real = phaseMLP.load_from_checkpoint("forward_real.ckpt")
        self.forward_Imag = phaseMLP.load_from_checkpoint("forward_imag.ckpt")
        self.forward_R = refMLP.load_from_checkpoint("forward_ref.ckpt")

        self.inverse_model = InverseMLP()

    def forward(self, x):
        # Forward models are not used in this forward pass; they are part of the tandem setup. [file:9]
        self.forward_Real.eval()
        self.forward_Imag.eval()
        self.forward_R.eval()
        return self.inverse_model(x)
