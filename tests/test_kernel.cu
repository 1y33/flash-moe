// Full kernel test in pure C++ — no Python, no torch, fast compile.
// Tests the real flash_moe_kernel with small random weights.
//
// Build:  make test_kernel
// Run:    ./build/test_kernel

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../csrc/kernel.cu"

int main() {
    namespace C = constants;

    printf("Config: H=%d I=%d E=%d K=%d\n", C::HIDDEN_SIZE, C::MOE_INTERMEDIATE_SIZE,
           C::NUM_EXPERTS, C::TOP_K);
    printf("Tiles: FFN1=%d FFN2=%d  Total tasks=%d  Capacity=%d\n",
           C::FFN1_TILES_PER_EXPERT, C::FFN2_TILES_PER_EXPERT, C::TOTAL_TASKS, C::CAPACITY);
    printf("Blocks: %d (1 OS + %d workers)\n\n", C::BLOCKSIZE, C::NUM_WORKERS);

    // Allocate model weights (small random)
    FlashMoe<float> model;
    allocate_flashmoe<float, CudaAllocator>(model);

    // Fill with random data via host
    srand(42);
    for (int e = 0; e < C::NUM_EXPERTS; e++) {
        float *h_gate, *h_up, *h_down;
        HostAllocator::allocate(&h_gate, C::GATE_PROJ_SIZE);
        HostAllocator::allocate(&h_up,   C::UP_PROJ_SIZE);
        HostAllocator::allocate(&h_down, C::DOWN_PROJ_SIZE);
        HostAllocator::fill_random(h_gate, C::GATE_PROJ_SIZE);
        HostAllocator::fill_random(h_up,   C::UP_PROJ_SIZE);
        HostAllocator::fill_random(h_down, C::DOWN_PROJ_SIZE);
        CudaAllocator::copy_to_device(h_gate, model.experts[e].gate_proj, C::GATE_PROJ_SIZE);
        CudaAllocator::copy_to_device(h_up,   model.experts[e].up_proj,   C::UP_PROJ_SIZE);
        CudaAllocator::copy_to_device(h_down, model.experts[e].down_proj, C::DOWN_PROJ_SIZE);
        HostAllocator::free(h_gate);
        HostAllocator::free(h_up);
        HostAllocator::free(h_down);
    }

    float *h_router;
    HostAllocator::allocate(&h_router, C::ROUTER_SIZE);
    HostAllocator::fill_random(h_router, C::ROUTER_SIZE);
    CudaAllocator::copy_to_device(h_router, model.router, C::ROUTER_SIZE);
    HostAllocator::free(h_router);

    // Build MoeState
    MoeState<float> state;
    CudaAllocator::allocate(&state.input,    C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.output,   C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.ffn1_out, C::TOP_K * C::MOE_INTERMEDIATE_SIZE);
    CudaAllocator::allocate(&state.ffn1_done, C::TOP_K);

    float *h_input;
    HostAllocator::allocate(&h_input, C::HIDDEN_SIZE);
    HostAllocator::fill_random(h_input, C::HIDDEN_SIZE);
    CudaAllocator::copy_to_device(h_input, state.input, C::HIDDEN_SIZE);
    HostAllocator::free(h_input);

    cudaMemset(state.output, 0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(float));

    int ffn1_init[C::TOP_K];
    for (int i = 0; i < C::TOP_K; i++)
        ffn1_init[i] = C::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);

    // Scheduling structures
    TaskQueue<C::CAPACITY> *task_queue;
    Doorbell *doorbells;
    int *status_queue;

    CudaAllocator::allocate(&task_queue, 1);
    CudaAllocator::allocate(&doorbells, C::NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, C::NUM_WORKERS);

    cudaMemset(task_queue, 0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells, 0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));

    printf("Launching kernel...\n");
    flash_moe_kernel<float><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read output
    float h_output[C::HIDDEN_SIZE];
    CudaAllocator::copy_to_host(state.output, h_output, C::HIDDEN_SIZE);

    float norm = 0.0f;
    for (int i = 0; i < C::HIDDEN_SIZE; i++)
        norm += h_output[i] * h_output[i];
    norm = sqrtf(norm);

    printf("Output norm: %.6f\n", norm);
    printf("Output[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);

    bool nonzero = norm > 1e-6f;
    printf("\n%s (output is %s)\n", nonzero ? "PASS" : "FAIL",
           nonzero ? "nonzero" : "all zeros — something went wrong");

    // Cleanup
    free_flashmoe<float, CudaAllocator>(model);
    CudaAllocator::free(state.input);
    CudaAllocator::free(state.output);
    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);

    return nonzero ? 0 : 1;
}
