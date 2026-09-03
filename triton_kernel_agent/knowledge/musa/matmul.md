### Matmul guidance

- Derive M, N, K, transposition, batching, and leading dimensions explicitly.
- Tile for data reuse, but keep register and shared-memory pressure within device limits.
- Coalesce global loads and make boundary predicates independent for M, N, and K.
- Accumulate low-precision inputs in the contractually required precision.
- Fuse epilogues only when they do not force unnecessary intermediate writes.
