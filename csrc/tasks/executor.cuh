#pragma once
#include <cuda_runtime.h>
#include "../flashmoe.cuh"
#include "../queue.cu"
#include "gemv.cuh"
#include "silu_mul.cuh"

template <typename T>
struct FFN1Executor {
    static __device__ __forceinline__
    void execute(Task &task, FlashMoe<T> *model,
                 T *input, float *ffn1_out,
                 DeviceTracer &tracer, long long *pending)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;
        bool is_lane0 = (threadIdx.x % 32 == 0);

        __shared__ float up_smem[constants::TILE_ROWS];

        // Fused gate+up: reads input x once, computes both projections
        if (is_lane0) tracer.start(TR_GEMV_GATE, pending);
        flashmoe::gemv_tile_fused_gate_up<T, 128>(
            model->experts[eid].gate_proj,
            model->experts[eid].up_proj,
            input,
            act,
            up_smem - task.row_begin,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);
        if (is_lane0) tracer.stop(TR_GEMV_GATE, pending);

        __syncthreads();

        if (is_lane0) tracer.start(TR_SILU_MUL, pending);
        for (int j = threadIdx.x; j < task.row_count; j += blockDim.x) {
            int r = task.row_begin + j;
            float g = act[r];
            float u = up_smem[j];
            act[r] = flashmoe::silu(g) * u;
        }
        if (is_lane0) tracer.stop(TR_SILU_MUL, pending);
    }

    static __device__ __forceinline__
    bool on_complete(Task &task, int *ffn1_done) {
        return atomicSub(&ffn1_done[task.slot], 1) == 1;
    }

    static __device__ __forceinline__
    void push_next(Task &task, TaskQueue<constants::CAPACITY> *task_queue) {
        int rows_left = constants::HIDDEN_SIZE;
        int row = 0;
        while (rows_left > 0) {
            int chunk = min(constants::TILE_ROWS, rows_left);
            Task t;
            t.expert_id = task.expert_id;
            t.weight    = task.weight;
            t.type      = FFN2;
            t.row_begin = row;
            t.row_count = chunk;
            t.slot      = task.slot;
            task_queue->push(t);
            row       += chunk;
            rows_left -= chunk;
        }
    }
};

// FFN2: down GEMV with accumulate for a row tile range
template <typename T>
struct FFN2Executor {
    static __device__ __forceinline__
    void execute(Task &task, FlashMoe<T> *model,
                 float *ffn1_out, float *output,
                 DeviceTracer &tracer, long long *pending)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;
        bool is_lane0 = (threadIdx.x % 32 == 0);

        if (is_lane0) tracer.start(TR_GEMV_DOWN, pending);
        flashmoe::gemv_tile_accumulate<float, 128>(
            reinterpret_cast<const float *>(model->experts[eid].down_proj), act, output,
            constants::MOE_INTERMEDIATE_SIZE,
            task.row_begin, task.row_count,
            task.weight);
        if (is_lane0) tracer.stop(TR_GEMV_DOWN, pending);
    }

    static __device__ __forceinline__
    void on_complete(Task &task) {
    }
};
