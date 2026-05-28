#include <cuda_runtime.h>
#include "flashmoe.cuh"
#include "tasks/gemv.cuh"
#include "utils/allocator.cuh"
#include "os.cu"
#include "worker.cu"

template <typename T, typename AccT>
__global__ __launch_bounds__(constants::THREADS_PER_BLOCK)
void flash_moe_kernel(
    FlashMoe<T> model,
    MoeState<T, AccT> state,
    TaskQueue<constants::CAPACITY> *task_queue,
    DeviceTracer tracer)
{
    namespace C = constants;
    __shared__ long long trace_pending[TRACE_MAX_WARPS];
    bool is_lane0 = (threadIdx.x % 32 == 0);

    constexpr int ROWS_PER_BLOCK = (C::NUM_EXPERTS + C::BLOCKSIZE - 1) / C::BLOCKSIZE;
    int my_row_begin = blockIdx.x * ROWS_PER_BLOCK;
    int my_row_count = min(ROWS_PER_BLOCK, C::NUM_EXPERTS - my_row_begin);

    if (is_lane0) tracer.start(TR_GEMV_ROUTE, trace_pending);
    if (my_row_count > 0)
    {
        flashmoe::gemv_tile<T, C::THREADS_PER_BLOCK>(
            model.router, state.input, state.logits,
            C::HIDDEN_SIZE, my_row_begin, my_row_count);
    }
    if (is_lane0) tracer.stop(TR_GEMV_ROUTE, trace_pending);

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(state.router_done, 1);
        __threadfence();
    }

    if (threadIdx.x == 0)
    {
        while (atomicAdd(state.router_done, 0) < C::BLOCKSIZE) {}
    }
    __syncthreads();

    if (blockIdx.x == 0)
    {
        OS<T>::run(state.logits, &model, task_queue, tracer, trace_pending);
    }
    else
    {
        int worker_id = blockIdx.x - 1;
        Worker<T, AccT>::run(worker_id, &model,
                       state.input, state.ffn1_out, state.output,
                       task_queue, state.ffn1_done, tracer, trace_pending);
    }
}

template <typename T, typename AccT>
void launch_flash_moe(T *input, float *output, FlashMoe<T> &model)
{
    namespace C = constants;

    MoeState<T, AccT> state;
    state.input = input;
    state.output = output;

    CudaAllocator::allocate(&state.ffn1_out, C::TOP_K * C::MOE_INTERMEDIATE_SIZE);
    CudaAllocator::allocate(&state.ffn1_done, C::TOP_K);
    CudaAllocator::allocate(&state.logits, C::NUM_EXPERTS);
    CudaAllocator::allocate(&state.router_done, 1);

    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(AccT));
    cudaMemset(output, 0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.logits, 0, C::NUM_EXPERTS * sizeof(float));
    cudaMemset(state.router_done, 0, sizeof(int));

    int ffn1_init[C::TOP_K];
    for (int i = 0; i < C::TOP_K; i++)
        ffn1_init[i] = C::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);

    TaskQueue<C::CAPACITY> *task_queue;
    CudaAllocator::allocate(&task_queue, 1);
    cudaMemset(task_queue, 0, sizeof(TaskQueue<C::CAPACITY>));

    // Trace buffer
    DeviceTracer tracer;
    tracer.max_events = 32768;
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    flash_moe_kernel<T, AccT><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, tracer);

    cudaDeviceSynchronize();

    TraceBuffer::print(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS);
    TraceBuffer::write_json(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS, "trace.json", TR_FIRST_LEAF);
    TraceBuffer::free(tracer.buf, tracer.count);

    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(state.logits);
    CudaAllocator::free(state.router_done);
    CudaAllocator::free(task_queue);
}

template void launch_flash_moe<float, float>(float *, float *, FlashMoe<float> &);
template void launch_flash_moe<__half, __half>(__half *, float *, FlashMoe<__half> &);
