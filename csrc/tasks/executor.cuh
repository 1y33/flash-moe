#pragma once
#include <cuda_runtime.h>
#include "../flashmoe.cuh"
#include "../queue.cu"
#include "gemv.cuh"
#include "silu_mul.cuh"

// FFN1: gate GEMV + up GEMV + silu_mul for a row tile range
// Writes activation into ffn1_out[slot * I + row_begin .. + row_count]
struct FFN1Executor {
    static __device__ __forceinline__
    void execute(Task &task, FlashMoe<float> *model,
                 float *input, float *ffn1_out)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;

        // gate GEMV: gate_buf[row_begin..+row_count] = gate_proj[row_begin:row_count, :] @ input
        flashmoe::gemv_tile<128>(
            model->experts[eid].gate_proj, input, act,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);

        // We need a separate buffer for up results before silu_mul.
        // Reuse the tail of ffn1_out as scratch (slot TOP_K is unused).
        float *up_scratch = ffn1_out + constants::TOP_K * constants::MOE_INTERMEDIATE_SIZE
                            + task.row_begin;

        flashmoe::gemv_tile<128>(
            model->experts[eid].up_proj, input, up_scratch - task.row_begin,
            constants::HIDDEN_SIZE,
            task.row_begin, task.row_count);

        __syncthreads();

        // silu_mul for our row range: act[r] = silu(gate[r]) * up[r]
        for (int j = threadIdx.x; j < task.row_count; j += blockDim.x) {
            int r = task.row_begin + j;
            float g = act[r];
            float u = up_scratch[j];
            act[r] = flashmoe::silu(g) * u;
        }
    }

    // Fan-in check + push FFN2 children
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
struct FFN2Executor {
    static __device__ __forceinline__
    void execute(Task &task, FlashMoe<float> *model,
                 float *ffn1_out, float *output)
    {
        int eid = task.expert_id;
        float *act = ffn1_out + task.slot * constants::MOE_INTERMEDIATE_SIZE;

        flashmoe::gemv_tile_accumulate<128>(
            model->experts[eid].down_proj, act, output,
            constants::MOE_INTERMEDIATE_SIZE,
            task.row_begin, task.row_count,
            task.weight);
    }

    static __device__ __forceinline__
    void on_complete(Task &task) {
        // nothing for now — could track total completion here
    }
};
