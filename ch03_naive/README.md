# Naive kernels

Reference code (read *after* writing yours): `../book.cu/1_naive`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] vector_add.cu, matrix_add.cu: elementwise
- [ ] transpose.cu
- [ ] gemm.cu: one thread per output element
- [ ] softmax.cu: naive, one thread per row
- [ ] conv1d.cu, conv2d.cu, maxpool2d.cu (3.3)

## Notes

Write the CPU version first, then the GPU one, and `check()` them against each other. That's the chapter's habit.
