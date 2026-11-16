### Common native MUSA constraints

- Keep real computation in `kernel.mu`; `binding.cpp` is binding and dispatch glue.
- Build with `torch_musa.utils.musa_extension.MUSAExtension` and `BuildExtension`.
- Expose `kernel_function` from `kernel.py` and keep the compiled module cached.
- Use `torch.musa` for events, synchronization, devices, and properties.
- Preserve input dtype, shape, stride, and device contracts. Do not silently call CUDA.
- Check every launch dimension and guard tail elements explicitly.
- Prefer simple, compilable source before architecture-specific intrinsics.
- Do not replace native computation with PyTorch eager operators.

### Recurring MUSA failure modes

- Template kernels cannot use `extern __shared__ scalar_t smem[];`.
  mcc reports: redeclaration of smem with a different type (double[] vs float[]).
  Declare the dynamic array with ONE concrete type and cast in the body.
- Never declare stride/padding/output_padding/groups/eps/scaling/output-size as kernel_function parameters.
  Bake them as module constants in kernel.py; bind_kernel_call supplies only model state and get_inputs().
- A reduction must cover the whole domain the reference op reduces over.
  GroupNorm/LayerNorm normalize over the full (channels_per_group, H, W) slab, not one row.
  One block per output row cannot compute those statistics: use a separate stats pass or one block per group.
- Keep every intermediate a later op reads: a residual add needs the PRE-norm conv output.
  Never overwrite that buffer with the normalized values.
