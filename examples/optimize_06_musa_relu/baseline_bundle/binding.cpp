#include <torch/extension.h>

void relu_musa_launch(const torch::Tensor& input, torch::Tensor& output);

namespace {

torch::Tensor forward(const torch::Tensor& input) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
              "input must have dtype torch.float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  auto output = torch::empty_like(input);
  if (input.numel() != 0) {
    relu_musa_launch(input, output);
  }
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Native MUSA relu forward (baseline bundle)");
}
