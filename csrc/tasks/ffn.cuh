#pragma once
#include <cuda_runtime.h>
#include "gemv.cuh"
#include "silu_mul.cuh"

namespace flashmoe
{

    template <int THREADS_PER_BLOCK = 128>
    __global__ void ffn1_kernel(
        const float *x,
        const float *gate_w,
        const float *up_w,
        float       *gate_buf,
        float       *up_buf,
        float       *act_out,
        int H, int I)
    {
        gemv_tile<float, THREADS_PER_BLOCK>(gate_w, x, gate_buf, H, 0, I);
        gemv_tile<float, THREADS_PER_BLOCK>(up_w,   x, up_buf,   H, 0, I);
        __syncthreads();
        silu_mul_tile<float, THREADS_PER_BLOCK>(gate_buf, up_buf, act_out, I);
    }

    template <int THREADS_PER_BLOCK = 128>
    __global__ void ffn2_kernel(
        const float *act,
        const float *down_w,
        float       *y,
        int H, int I,
        float scale)
    {
        gemv_tile_accumulate<float, THREADS_PER_BLOCK>(
            down_w, act, y, I, 0, H, scale);
    }

} // namespace flashmoe
