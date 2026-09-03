"""Small native-MUSA optimization target used by the MCU example."""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


# MCU 1.3 samples hardware counters periodically.  A 256K-element sigmoid only
# runs for about 8 us on MUSA and falls between the profiler's overlap/overflow
# limits.  Use a presentation-sized workload so live SOL collection is stable
# and the measured optimization difference is meaningful.
NUM_ELEMENTS = 4 * 1024 * 1024


def get_inputs():
    # Inputs stay on CPU here; the benchmark/test harness moves them to MUSA.
    return [torch.randn(NUM_ELEMENTS, dtype=torch.float32)]


def get_init_inputs():
    return []
