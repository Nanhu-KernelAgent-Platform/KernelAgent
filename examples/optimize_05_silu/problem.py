# SiLU (Swish) Activation — simple element-wise GPU kernel example
#
# Designed to be simpler than RMSNorm (no reduction, no normalization).
# The forward pass is purely element-wise:  y = x * sigmoid(x)
# This is the default activation in modern LLMs such as Mistral and LLaMA.

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """Simple model that applies the SiLU (Swish) activation function."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SiLU activation:  SiLU(x) = x * sigmoid(x)

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SiLU applied, same shape as input.
        """
        return x * torch.sigmoid(x)


# Keep the original benchmark shape by default while allowing profiling and
# live demos to use a smaller tensor without editing this problem file.
batch_size = int(os.getenv("SILU_BATCH_SIZE", "4096"))
seq_len = int(os.getenv("SILU_SEQ_LEN", "4096"))


def get_inputs():
    x = torch.rand(batch_size, seq_len)
    return [x]


def get_init_inputs():
    return []
