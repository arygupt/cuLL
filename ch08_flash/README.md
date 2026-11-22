# Flash Attention

Reference code (read *after* writing yours): `../book.cu/6_flash`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] naive_attention.cu: S = QKᵀ, softmax, then ·V, materializing S
- [ ] flash_attention.cu: tiled with online softmax, never materializing S (8.3)

## Notes

The repo's fa.cu uses WMMA. You can write the tiled/online-softmax version with plain FP32 CUDA cores first, which runs at full speed on your card, then try the WMMA version.
