### Transpose guidance

- Use tiled shared-memory staging to turn both reads and writes into coalesced accesses.
- Pad shared-memory rows when necessary to avoid bank conflicts.
- Separate logical shape, physical stride, and permutation semantics.
- Handle rectangular and non-multiple tile sizes with independent predicates.
- For packed or FP8 data, preserve byte-level layout and scaling semantics.
