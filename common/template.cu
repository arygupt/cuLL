// Starting point for a new kernel. Copy it into a chapter folder:
//   cp ../common/template.cu softmax.cu
// then write the CPU reference and the kernel, and run:  book make run-softmax
#include "util.cuh"
#include <vector>

// 1. CPU reference: the simplest correct version. Your GPU result is checked against this.
void cpu_ref(const float* in, float* out, int n) {
  // TODO
}

// 2. Your kernel.
__global__ void my_kernel(const float* in, float* out, int n) {
  // TODO
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_in(n), h_out(n), h_ref(n);
  fill_random(h_in.data(), n);
  cpu_ref(h_in.data(), h_ref.data(), n);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  const int block = 256, grid = ceil_div(n, block);
  my_kernel<<<grid, block>>>(d_in, d_out, n);
  KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  check(h_out.data(), h_ref.data(), n);

  // 3. Predict first, then measure. Count the bytes your kernel reads + writes.
  float ms = time_ms([&] { my_kernel<<<grid, block>>>(d_in, d_out, n); });
  report("my_kernel", ms, 2.0 * bytes);

  cudaFree(d_in); cudaFree(d_out);
}
