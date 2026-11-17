#include "vignette_kernel.h"

#include <cuda_runtime.h>

struct VcKernel {
  float *table;  /* device memory, (VC_TABLE_SIZE,) float32 */
  /* Own stream, so waiting for the kernel does not also wait for other GPU
   * work in the process, such as nvinfer's inference. */
  cudaStream_t stream;
};

static bool cuda_ok(cudaError_t status, const char *what, std::string *error)
{
  if (status == cudaSuccess)
    return true;
  if (error)
    *error = std::string(what) + ": " + cudaGetErrorString(status);
  return false;
}

/* rgba: (height, pitch) uint8 with 4 bytes per pixel, corrected in place. */
__global__ void vignette_correct_kernel(uint8_t *rgba, int pitch, int width, int height,
                                        const float *table, VcParams params)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  /* One 32-bit load and at most one store per pixel; rows are 4-byte aligned. */
  uint32_t *px = (uint32_t *) (rgba + (size_t) y * pitch) + x;
  uint32_t in = *px;
  uint32_t out = vc_correct_word(in, vc_gain(table, &params, x, y), &params);
  if (out != in)
    *px = out;
}

VcKernel *vc_kernel_create(const float *table, std::string *error)
{
  VcKernel *kernel = new VcKernel();
  kernel->table = NULL;
  kernel->stream = NULL;
  size_t bytes = VC_TABLE_SIZE * sizeof(float);
  if (!cuda_ok(cudaStreamCreate(&kernel->stream), "cudaStreamCreate", error) ||
      !cuda_ok(cudaMalloc(&kernel->table, bytes), "cudaMalloc", error) ||
      !cuda_ok(cudaMemcpy(kernel->table, table, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy", error)) {
    vc_kernel_destroy(kernel);
    return NULL;
  }
  return kernel;
}

bool vc_kernel_run(VcKernel *kernel, uint8_t *rgba, int pitch, int width, int height,
                   const VcParams &params, std::string *error)
{
  if (pitch % 4 != 0 || ((uintptr_t) rgba) % 4 != 0) {
    if (error)
      *error = "frame rows are not 4-byte aligned";
    return false;
  }
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  vignette_correct_kernel<<<grid, block, 0, kernel->stream>>>(rgba, pitch, width, height,
                                                             kernel->table, params);
  if (!cuda_ok(cudaGetLastError(), "kernel launch", error))
    return false;
  return cuda_ok(cudaStreamSynchronize(kernel->stream), "kernel run", error);
}

void vc_kernel_destroy(VcKernel *kernel)
{
  if (!kernel)
    return;
  if (kernel->table)
    cudaFree(kernel->table);
  if (kernel->stream)
    cudaStreamDestroy(kernel->stream);
  delete kernel;
}
