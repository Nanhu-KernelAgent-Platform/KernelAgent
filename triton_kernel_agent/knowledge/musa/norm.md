### Normalization guidance

- Compute sum/sum-of-squares in fp32 for fp16 and bf16 inputs.
- Reuse loaded values when register pressure permits; otherwise use a two-stage design.
- Treat epsilon, affine weight, residual, and bias as part of the public contract.
- Optimize small and large normalized dimensions separately when their regimes differ.
- Validate odd sizes and non-power-of-two tails.
- Statistics domain: GroupNorm reduces over (channels_per_group, H, W) per sample.
  A block that owns one output row sees only W values and cannot compute it.
  Do a statistics pass first, then normalize; keep the pre-norm tensor for residual adds.
