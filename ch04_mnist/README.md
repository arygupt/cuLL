# Training an MLP in raw CUDA

Reference code (read *after* writing yours): `../book.cu/2_mnist`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] Run the levels from the repo first: `cd ../book.cu/2_mnist && book uv run v0.py` (downloads data), then v1.py, v2.py
- [ ] mnist.cu: your version of v4 (naive CUDA forward + backward)
- [ ] mnist_cublas.cu: your version of v5 (matmuls via cuBLAS)

## Notes

MNIST data is downloaded by `v0.py` into `book.cu/2_mnist` on the PC; point your code at `../book.cu/2_mnist/data`.
