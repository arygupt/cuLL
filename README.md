# cuLL: working through *CUDA for LLMs*

Write kernels here on the Mac; `book` runs them on the PC's GTX 1660 (WSL Ubuntu, CUDA 13.1) over Tailscale.

```
CUDA_for_LLMs.pdf / .epub   the book
book.cu/                    the book's reference code (read after writing yours)
common/                     util.cuh (CUDA_CHECK, timing, check, report), template.cu, chapter.mk
ch02_first_program/ …       your kernels, one folder per chapter; README.md in each = checklist
LAB.md                      your measurement log
```

## Loop for each kernel

```bash
cd ch06_optim/gemv
cp ../../common/template.cu 2_coalesced_warp.cu   # write the CPU reference + kernel
book make run-2_coalesced_warp                     # syncs to the PC, builds for sm_75, runs
book ncu --set full ./2_coalesced_warp             # profile
```

- `book make`: build every `.cu` in the folder · `book make clean`
- `book <anything>`: runs that command on the PC in the matching folder (`book nvidia-smi`, `book uv run v1.py`)
- `book`: interactive shell on the PC · `book --pull file`: copy a file back to the Mac
- Debug build: `book make foo NVCC_FLAGS="-O0 -G"` then `book compute-sanitizer ./foo`

## GTX 1660 numbers (for "% of peak")

sm_75 Turing · 6 GB · **~192 GB/s** memory · **~5 TFLOPS** FP32 · fast FP16 · INT8 `dp4a` · **no tensor cores, no BF16**

Chapters 2–6 and 9 run fully. Chapter 7 (WMMA) and 8 (flash) run correctly but slowly. WGMMA, CUTLASS (ch 11) and distributed (ch 10) need rented H100s.

## Editor

Neovim with clangd (via Mason) gives CUDA autocomplete and errors. `.clangd` here points it at Linux and CUDA headers copied into `.sysroot/`. Restart nvim (or `:LspRestart`) if it's already open.
