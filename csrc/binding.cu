#include <torch/extension.h>
#include <cuda_fp16.h>
#include "flashmoe.cuh"
#include "utils/allocator.cuh"
#include "queue.cu"

template <typename T, typename AccT>
void launch_flash_moe(T *input, float *output, FlashMoe<T> &model);

torch::Tensor flash_moe_forward(
    torch::Tensor input,
    torch::Tensor router_weight,
    std::vector<torch::Tensor> gate_projs,
    std::vector<torch::Tensor> up_projs,
    std::vector<torch::Tensor> down_projs)
{
    auto output = torch::zeros({constants::HIDDEN_SIZE},
                               torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    if (input.scalar_type() == torch::kFloat16)
    {
        FlashMoe<__half> model;
        model.router = reinterpret_cast<__half *>(router_weight.data_ptr<at::Half>());
        for (int e = 0; e < constants::NUM_EXPERTS; e++)
        {
            model.experts[e].gate_proj = reinterpret_cast<__half *>(gate_projs[e].data_ptr<at::Half>());
            model.experts[e].up_proj = reinterpret_cast<__half *>(up_projs[e].data_ptr<at::Half>());
            model.experts[e].down_proj = reinterpret_cast<__half *>(down_projs[e].data_ptr<at::Half>());
        }
        launch_flash_moe<__half, __half>(
            reinterpret_cast<__half *>(input.data_ptr<at::Half>()),
            output.data_ptr<float>(), model);
    }
    else
    {
        FlashMoe<float> model;
        model.router = router_weight.data_ptr<float>();
        for (int e = 0; e < constants::NUM_EXPERTS; e++)
        {
            model.experts[e].gate_proj = gate_projs[e].data_ptr<float>();
            model.experts[e].up_proj = up_projs[e].data_ptr<float>();
            model.experts[e].down_proj = down_projs[e].data_ptr<float>();
        }
        launch_flash_moe<float, float>(input.data_ptr<float>(), output.data_ptr<float>(), model);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &flash_moe_forward, "FlashMoE forward");
}
