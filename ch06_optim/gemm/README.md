# Optimizing: GEMM (6.6)

Reference code (read *after* writing yours): `../../book.cu/4_optim/gemm`

Start a kernel: `cp ../../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] 1_naive.cu
- [ ] 2_gmem_coalesce.cu
- [ ] 3_smem_blocking.cu: shared-memory tiles
- [ ] 4_1d_blocktiling.cu: each thread computes several outputs
- [ ] 5_2d_blocktiling.cu
- [ ] 6_vectorize.cu
- [ ] Compare against cuBLAS each step (`cublasSgemm`). To run the repo's version: `cd ../../book.cu/4_optim/gemm && book make ARCH=sm_75`

## Notes

GEMM is compute-bound at large sizes: report GFLOP/s (2·M·N·K flops) vs ~5000 peak.
