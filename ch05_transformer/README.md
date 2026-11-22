# Transformer inference in CUDA

Reference code (read *after* writing yours): `../book.cu/3_transformer`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] Read through the repo's csrc/ and wrapper/ first: it builds a PyTorch extension (`book uv pip install -e ../book.cu/3_transformer`)
- [ ] Write the inference kernels from 5.3 standalone here first (one .cu each), checked against PyTorch or a CPU reference
- [ ] Then swap your kernels into a copy of the extension

## Notes

`TORCH_CUDA_ARCH_LIST=7.5` is already set on the PC so extensions build for your card.
