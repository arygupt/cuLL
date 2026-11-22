# Optimizing: GEMV (6.2)

Reference code (read *after* writing yours): `../../book.cu/4_optim/gemv`

Start a kernel: `cp ../../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] 1_naive.cu
- [ ] 2_coalesced_warp.cu: one warp per row, warp-shuffle reduction
- [ ] 3_coalesced_warpblock.cu
- [ ] 4_vectorized.cu: float4 loads
- [ ] Compare each against cuBLAS (0_cublas in the repo). Target: close to 192 GB/s

## Notes

GEMV is memory-bound: the number that matters is GB/s vs the 192 GB/s peak.
