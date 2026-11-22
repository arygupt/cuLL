// Shared helpers for your own kernels: error checking, timing, comparing results, random data.
// Include with:  #include "util.cuh"   (the Makefiles add -I../common)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Wrap every CUDA API call: CUDA_CHECK(cudaMalloc(&d, bytes));
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n  %s\n", cudaGetErrorName(err_), \
              __FILE__, __LINE__, cudaGetErrorString(err_));                    \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// After a kernel launch: my_kernel<<<g, b>>>(...); KERNEL_CHECK();
#define KERNEL_CHECK()                     \
  do {                                     \
    CUDA_CHECK(cudaGetLastError());        \
    CUDA_CHECK(cudaDeviceSynchronize());   \
  } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Fill with uniform random floats in [lo, hi). Same seed → same data.
inline void fill_random(float* x, size_t n, float lo = -1.f, float hi = 1.f, unsigned seed = 42) {
  srand(seed);
  for (size_t i = 0; i < n; i++) x[i] = lo + (hi - lo) * (rand() / (float)RAND_MAX);
}

// Compare a GPU result against a CPU reference. Prints the worst element; returns true if all within tol.
inline bool check(const float* got, const float* want, size_t n, float tol = 1e-4f, const char* name = "result") {
  size_t worst = 0; float worst_err = 0.f;
  for (size_t i = 0; i < n; i++) {
    float err = fabsf(got[i] - want[i]) / fmaxf(1.f, fabsf(want[i]));
    if (!(err <= worst_err)) { worst_err = err; worst = i; }  // also catches NaN
  }
  bool ok = worst_err <= tol;
  printf("%s %s: max rel err %.2e at [%zu] (got %g, want %g)\n", ok ? "PASS" : "FAIL", name,
         worst_err, worst, got[worst], want[worst]);
  return ok;
}

// Time a kernel with CUDA events: does `warmup` untimed runs, then returns average ms over `iters`.
//   float ms = time_ms([&]{ my_kernel<<<g, b>>>(...); });
template <typename F>
float time_ms(F launch, int iters = 20, int warmup = 3) {
  for (int i = 0; i < warmup; i++) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return ms / iters;
}

// GTX 1660 peaks, for "% of peak" in your lab log.
constexpr double PEAK_GBPS   = 192.0;   // GDDR5, 192-bit @ 8 Gbps
constexpr double PEAK_GFLOPS = 5000.0;  // ~5 TFLOPS FP32 (1408 cores × 2 × ~1.785 GHz boost)

// Print time, bandwidth and throughput for one run. Pass bytes moved and flops done by the kernel.
inline void report(const char* name, float ms, double bytes, double flops = 0) {
  double gbps = bytes / ms / 1e6, gflops = flops / ms / 1e6;
  printf("%-28s %8.3f ms  %7.1f GB/s (%4.1f%% peak)", name, ms, gbps, 100 * gbps / PEAK_GBPS);
  if (flops > 0) printf("  %8.1f GFLOP/s (%4.1f%% peak)", gflops, 100 * gflops / PEAK_GFLOPS);
  printf("\n");
}
