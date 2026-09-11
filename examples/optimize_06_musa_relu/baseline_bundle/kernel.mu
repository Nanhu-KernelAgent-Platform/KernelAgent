#include <ATen/musa/MUSAContext.h>
#include <musa_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void relu_float4_kernel(const float4* __restrict__ input,
                                   float4* __restrict__ output,
                                   int64_t vector_count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vector_count) {
    return;
  }

  const float4 value = input[index];
  output[index] = make_float4(fmaxf(value.x, 0.0f), fmaxf(value.y, 0.0f),
                              fmaxf(value.z, 0.0f), fmaxf(value.w, 0.0f));
}

__global__ void relu_scalar_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = fmaxf(input[index], 0.0f);
  }
}

}  // namespace

void relu_musa_launch(const torch::Tensor& input, torch::Tensor& output) {
  constexpr int threads = 256;
  const int64_t count = input.numel();
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  if ((count & 3) == 0) {
    const int64_t vector_count = count / 4;
    const int blocks = static_cast<int>((vector_count + threads - 1) / threads);
    relu_float4_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()), vector_count);
  } else {
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relu_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), count);
  }

  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "MUSA relu kernel launch failed: ",
              musaGetErrorString(error));
}
