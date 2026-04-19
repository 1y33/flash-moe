#include <cuda_runtime.h>
#include "flashmoe.cuh"
#include "allocator.cu"
#include "os.cu"
#include "worker.cu"

__global__ void flash_moe_kernel(
    FlashMoe<float> model,
    MoeState<float> state,
    TaskQueue<constants::CAPACITY> *task_queue,
    Doorbell *doorbells,
    int *status_queue)
{
    if (blockIdx.x == 0) {
        OS::run(state.input, &model, task_queue, doorbells,
                status_queue, state.ffn1_done,
                constants::NUM_WORKERS, constants::TOTAL_TASKS);
    }
    else {
        int worker_id = blockIdx.x - 1;
        Worker::run(worker_id, &model,
                    state.input, state.ffn1_out, state.output,
                    task_queue, doorbells, status_queue, state.ffn1_done);
    }
}

void launch_flash_moe(float *input, float *output, FlashMoe<float> &model) {
    namespace C = constants;

    MoeState<float> state;
    state.input = input;
    state.output = output;

    CudaAllocator::allocate(&state.ffn1_out, C::TOP_K * C::MOE_INTERMEDIATE_SIZE);
    CudaAllocator::allocate(&state.ffn1_done, C::TOP_K);

    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(float));
    cudaMemset(output, 0, C::HIDDEN_SIZE * sizeof(float));

    int ffn1_init[C::TOP_K];
    for (int i = 0; i < C::TOP_K; i++)
        ffn1_init[i] = C::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);

    TaskQueue<C::CAPACITY> *task_queue;
    Doorbell *doorbells;
    int *status_queue;

    CudaAllocator::allocate(&task_queue, 1);
    CudaAllocator::allocate(&doorbells, C::NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, C::NUM_WORKERS);

    cudaMemset(task_queue, 0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells, 0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));

    flash_moe_kernel<<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue);

    cudaDeviceSynchronize();

    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);
}
