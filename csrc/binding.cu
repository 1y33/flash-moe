#include <torch/extension.h>
#include "flashmoe.cuh"
#include "allocator.cu"
#include "queue.cu"

void launch_flash_moe(float *input, float *output, FlashMoe<float> &model);

torch::Tensor flash_moe_forward(
    torch::Tensor input,
    torch::Tensor router_weight,
    std::vector<torch::Tensor> gate_projs,
    std::vector<torch::Tensor> up_projs,
    std::vector<torch::Tensor> down_projs)
{
    FlashMoe<float> model;
    model.router = router_weight.data_ptr<float>();
    for (int e = 0; e < constants::NUM_EXPERTS; e++)
    {
        model.experts[e].gate_proj = gate_projs[e].data_ptr<float>();
        model.experts[e].up_proj = up_projs[e].data_ptr<float>();
        model.experts[e].down_proj = down_projs[e].data_ptr<float>();
    }

    auto output = torch::zeros({constants::HIDDEN_SIZE}, input.options());
    launch_flash_moe(input.data_ptr<float>(), output.data_ptr<float>(), model);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &flash_moe_forward, "FlashMoE forward");
}
