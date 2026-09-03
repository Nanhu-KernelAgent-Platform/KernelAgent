### Elementwise guidance

- Flatten contiguous regions and use grid-stride loops for broad shape coverage.
- Vectorize only when pointer alignment and tail handling are explicit.
- Fuse adjacent cheap operations to reduce global-memory traffic.
- Avoid expensive control-flow divergence inside a warp.
- Preserve broadcasting and non-contiguous-input semantics required by the contract.
