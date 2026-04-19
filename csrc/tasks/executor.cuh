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
                 T *input, float *ffn1_out)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;

        __shared__ float up_smem[constants::TILE_ROWS];

        flashmoe::gemv_tile<T, 128>(
            model->experts[eid].gate_proj, input, act,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);

        flashmoe::gemv_tile<T, 128>(
            model->experts[eid].up_proj, input, up_smem - task.row_begin,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);

        __syncthreads();

        // silu(gate) * up, in-place on act
        for (int j = threadIdx.x; j < task.row_count; j += blockDim.x) {
            int r = task.row_begin + j;
            float g = act[r];
            float u = up_smem[j];
            act[r] = flashmoe::silu(g) * u;
        }
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
// output[row_begin..+row_count] += weight * down_proj[row_begin:row_count, :] @ act
// down_proj is T, act (ffn1_out) is float, output is float
template <typename T>
struct FFN2Executor {
    static __device__ __forceinline__
    void execute(Task &task, FlashMoe<T> *model,
                 float *ffn1_out, float *output)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;

        // TODO: when T != float, need mixed-type GEMV (A=T, x=float)
        // For now works correctly when T=float
        flashmoe::gemv_tile_accumulate<float, 128>(
            reinterpret_cast<const float *>(model->experts[eid].down_proj), act, output,
            constants::MOE_INTERMEDIATE_SIZE,
            task.row_begin, task.row_count,
            task.weight);
    }

    static __device__ __forceinline__
    void on_complete(Task &task) {
    }
};
