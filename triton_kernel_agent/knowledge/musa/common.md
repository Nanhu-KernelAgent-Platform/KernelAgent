### Common native MUSA constraints

- Keep real computation in `kernel.mu`; `binding.cpp` is binding and dispatch glue.
- Build with `torch_musa.utils.musa_extension.MUSAExtension` and `BuildExtension`.
- Expose `kernel_function` from `kernel.py` and keep the compiled module cached.
- Use `torch.musa` for events, synchronization, devices, and properties.
- Preserve input dtype, shape, stride, and device contracts. Do not silently call CUDA.
- Check every launch dimension and guard tail elements explicitly.
- Prefer simple, compilable source before architecture-specific intrinsics.
- Do not replace native computation with PyTorch eager operators.
