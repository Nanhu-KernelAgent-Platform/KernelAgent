### MoE and grouped-GEMM guidance

- Preserve token-to-expert routing, offsets, expert order, and empty-expert behavior.
- Bucket or specialize strongly different expert sizes rather than forcing one tile.
- Avoid host synchronization and repeated allocation in the dispatch path.
- Make grouped metadata compact and device-resident when the contract permits.
- Validate zero-token experts, uneven groups, transposed weights, and fused epilogues.
