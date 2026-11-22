# Tensor cores (WMMA only)

Reference code (read *after* writing yours): `../book.cu/5_tensor_cores`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] wmma_gemm.cu: 16×16×16 fp16 tiles with `nvcuda::wmma` (7.2)
- [ ] WGMMA (7.3) and TCGen05 (7.7) need an H100/B200: read them, run on a rented GPU

## Notes

**Your GTX 1660 has no tensor cores.** WMMA compiles and gives correct answers but runs slowly (~0.8 TFLOPS). Check correctness here; save speed comparisons for a rented GPU.
