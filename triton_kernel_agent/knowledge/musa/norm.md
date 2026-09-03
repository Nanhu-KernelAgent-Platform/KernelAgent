### Normalization guidance

- Compute sum/sum-of-squares in fp32 for fp16 and bf16 inputs.
- Reuse loaded values when register pressure permits; otherwise use a two-stage design.
- Treat epsilon, affine weight, residual, and bias as part of the public contract.
- Optimize small and large normalized dimensions separately when their regimes differ.
- Validate odd sizes and non-power-of-two tails.
