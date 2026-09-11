"""Native-MUSA optimization target: elementwise ReLU.

Used by the kernelagent-injector integration test for the MNIST training script
(train_mnist_musa.py) where F.relu is the operator being replaced.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


# Presentation-size workload so MCU profiling samples stay stable, mirroring
# optimize_04_musa_sigmoid.
NUM_ELEMENTS = 4 * 1024 * 1024


def get_inputs():
    # Inputs stay on CPU here; the benchmark/test harness moves them to MUSA.
    return [torch.randn(NUM_ELEMENTS, dtype=torch.float32)]


def get_init_inputs():
    return []
