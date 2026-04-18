#include <cuda_runtime.h>
#include "flashmoe.cuh"
#include "allocator.cu"
#include "os.cu"
#include "worker.cu"

__global__ void flash_moe_kernel(
    float *input,
    float *output,
    float *ffn1_out,
    FlashMoe<float> model,
    TaskQueue<constants::CAPACITY> *task_queue,
    Doorbell *doorbells,
    int *status_queue,
    int *ffn1_done)
{
    if (blockIdx.x == 0)
    {
        OS::run(input, &model, task_queue, doorbells,
                status_queue, ffn1_done,
                constants::NUM_WORKERS, constants::TOTAL_TASKS);
    }
    else
    {
        int worker_id = blockIdx.x - 1;
        Worker::run(worker_id, &model, input, ffn1_out, output,
                    task_queue, doorbells, status_queue, ffn1_done);
    }
}

void launch_flash_moe(float *input, float *output, FlashMoe<float> &model)
{
    float *ffn1_out;
    CudaAllocator::allocate(&ffn1_out,
                            (constants::TOP_K + 1) * constants::MOE_INTERMEDIATE_SIZE);
    cudaMemset(ffn1_out, 0,
               (constants::TOP_K + 1) * constants::MOE_INTERMEDIATE_SIZE * sizeof(float));

    TaskQueue<constants::CAPACITY> *task_queue;
    Doorbell *doorbells;
    int *status_queue;
    int *ffn1_done;

    CudaAllocator::allocate(&task_queue, 1);
    CudaAllocator::allocate(&doorbells, constants::NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, constants::NUM_WORKERS);
    CudaAllocator::allocate(&ffn1_done, constants::TOP_K);

    cudaMemset(task_queue, 0, sizeof(TaskQueue<constants::CAPACITY>));
    cudaMemset(doorbells, 0, constants::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, constants::NUM_WORKERS * sizeof(int));

    int ffn1_init[constants::TOP_K];
    for (int i = 0; i < constants::TOP_K; i++)
        ffn1_init[i] = constants::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, ffn1_done, constants::TOP_K);

    cudaMemset(output, 0, constants::HIDDEN_SIZE * sizeof(float));

    flash_moe_kernel<<<constants::BLOCKSIZE, constants::THREADS_PER_BLOCK>>>(
        input, output, ffn1_out, model,
        task_queue, doorbells, status_queue, ffn1_done);

    cudaDeviceSynchronize();

    CudaAllocator::free(ffn1_out);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);
    CudaAllocator::free(ffn1_done);
}
