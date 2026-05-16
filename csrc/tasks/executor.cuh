#pragma once
#include <cuda_runtime.h>
#include "../flashmoe.cuh"
#include "../queue.cu"
#include "../utils/dtypes.cuh"
#include "gemv_ffn1.cuh"
#include "gemv_ffn2.cuh"
#include "silu_mul.cuh"

template <typename T, typename AccT = __half>
struct FFN1Executor
{
    using DA = DType<AccT>;

    static __device__ __forceinline__ void execute(Task &task, FlashMoe<T> *model,
                                                   T *input, AccT *ffn1_out,
                                                   DeviceTracer &tracer, long long *pending)
    {
        int eid = task.expert_id;
        AccT *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;
        bool is_lane0 = (threadIdx.x % 32 == 0);

        __shared__ AccT up_smem[constants::TILE_ROWS];

        if (is_lane0)
            tracer.start(TR_GEMV_GATE, pending);
        flashmoe::gemv_tile_fused_gate_up_prefetch<T, AccT, constants::THREADS_PER_BLOCK>(
            model->experts[eid].gate_proj,
            model->experts[eid].up_proj,
            input,
            act,
            up_smem - task.row_begin,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);
        if (is_lane0)
            tracer.stop(TR_GEMV_GATE, pending);

        __syncthreads();

        if (is_lane0)
            tracer.start(TR_SILU_MUL, pending);
        for (int j = threadIdx.x; j < task.row_count; j += blockDim.x)
        {
            float g = DA::load(&act[task.row_begin + j]);
            float u = DA::load(&up_smem[j]);
            DA::store(&act[task.row_begin + j], flashmoe::silu_fast(g) * u);
        }
        if (is_lane0)
            tracer.stop(TR_SILU_MUL, pending);
    }

    // Decrements the fan-in counter. FFN2 workers spin on this counter
    // hitting 0 before they read activations.
    static __device__ __forceinline__ void on_complete(Task &task, int *ffn1_done)
    {
        atomicSub(&ffn1_done[task.slot], 1);
    }
};

template <typename T, typename AccT = __half>
struct FFN2Executor
{
    static __device__ __forceinline__ void execute(Task &task, FlashMoe<T> *model,
                                                   AccT *ffn1_out, float *output,
                                                   int *ffn1_done,
                                                   DeviceTracer &tracer, long long *pending)
    {
        int eid = task.expert_id;
        AccT *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;
        bool is_lane0 = (threadIdx.x % 32 == 0);

        // Wait until all FFN1 tiles for this slot have completed. Since FFN2
        // tasks are pre-pushed by the bootstrap dispatcher, a worker may pick
        // one up before its activations are ready.
        if (threadIdx.x == 0)
        {
            while (atomicAdd(&ffn1_done[task.slot], 0) != 0)
            {
            }
            __threadfence();
        }
        __syncthreads();

        if (is_lane0)
            tracer.start(TR_GEMV_DOWN, pending);
        flashmoe::gemv_tile_accumulate_mixed_prefetch<T, AccT, constants::THREADS_PER_BLOCK>(
            model->experts[eid].down_proj, act, output,
            constants::MOE_INTERMEDIATE_SIZE,
            task.row_begin, task.row_count,
            task.weight);
        if (is_lane0)
            tracer.stop(TR_GEMV_DOWN, pending);
    }

    static __device__ __forceinline__ void on_complete(Task &task)
    {
    }
};
