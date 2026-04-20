#include <cuda_runtime.h>
#include "flashmoe.cuh"
#include "utils/allocator.cuh"
#include "os.cu"
#include "worker.cu"

template <typename T>
__global__ void flash_moe_kernel(
    FlashMoe<T> model,
    MoeState<T> state,
    TaskQueue<constants::CAPACITY> *task_queue,
    Doorbell *doorbells,
    int *status_queue,
    DeviceTracer tracer)
{
    __shared__ long long trace_pending[TRACE_MAX_WARPS];

    if (blockIdx.x == 0)
    {
        OS<T>::run(state.input, &model, task_queue, doorbells,
                   status_queue, state.ffn1_done,
                   constants::NUM_WORKERS, constants::TOTAL_TASKS, tracer, trace_pending);
    }
    else
    {
        int worker_id = blockIdx.x - 1;
        Worker<T>::run(worker_id, &model,
                       state.input, state.ffn1_out, state.output,
                       task_queue, doorbells, status_queue, state.ffn1_done, tracer, trace_pending);
    }
}

template <typename T>
void launch_flash_moe(T *input, float *output, FlashMoe<T> &model)
{
    namespace C = constants;

    MoeState<T> state;
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

    // Trace buffer
    DeviceTracer tracer;
    tracer.max_events = 32768;
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    flash_moe_kernel<T><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue, tracer);

    cudaDeviceSynchronize();

    TraceBuffer::print(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS);
    TraceBuffer::write_json(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS, "trace.json", TR_FIRST_LEAF);
    TraceBuffer::free(tracer.buf, tracer.count);

    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);
}

template void launch_flash_moe<float>(float *, float *, FlashMoe<float> &);
template void launch_flash_moe<__half>(__half *, float *, FlashMoe<__half> &);
