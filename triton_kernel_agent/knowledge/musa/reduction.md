### Reduction guidance

- Identify the exact reduction axes and whether they are contiguous.
- Use hierarchical block reduction rather than global atomics when practical.
- Accumulate fp16/bf16 values in fp32 when the numerical contract requires it.
- Size shared memory from the actual block configuration and guard partial tiles.
- For softmax, combine max and sum reductions carefully and preserve stability.
