#pragma once
#include <cuda_runtime.h>

namespace flashmoe
{

    template <int WARP_SIZE = 32>
    __device__ __forceinline__ float warp_reduce_sum(float v)
    {
        v += __shfl_down_sync(0xffffffff, v, 16);
        v += __shfl_down_sync(0xffffffff, v, 8);
        v += __shfl_down_sync(0xffffffff, v, 4);
        v += __shfl_down_sync(0xffffffff, v, 2);
        v += __shfl_down_sync(0xffffffff, v, 1);
        return v;
    }

    template <int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void gemv_tile(
        const float *__restrict__ A,
        const float *__restrict__ x,
        float *__restrict__ y,
        int N,
        int row_begin,
        int row_count)
    {
        constexpr int WARP_SIZE = 32;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x & 31;

        const int N4 = N >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);

        // Strided row assignment: warp w handles rows w, w+WARPS, w+2*WARPS, ...
        for (int r = warp_id; r < row_count; r += WARPS_PER_BLOCK)
        {
            int row = row_begin + r;
            const float4 *A4 = reinterpret_cast<const float4 *>(A + (size_t)row * N);

            float acc = 0.0f;

            // Each iteration does 4 FMAs on 16 bytes of A and 16 bytes of x.
            #pragma unroll 4
            for (int j = lane; j < N4; j += WARP_SIZE)
            {
                float4 a = A4[j];
                float4 b = x4[j];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }

            acc = warp_reduce_sum(acc);
            if (lane == 0) y[row] = acc;
        }
    }

    // Variant that ACCUMULATES into y (y += A*x) — used later for the final
    // gather step where we sum top-k expert outputs weighted by the router.
    // `scale` is the router weight for this expert.
    template <int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void gemv_tile_accumulate(
        const float *__restrict__ A,
        const float *__restrict__ x,
        float *__restrict__ y,
        int N,
        int row_begin,
        int row_count,
        float scale)
    {
        constexpr int WARP_SIZE       = 32;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

        const int warp_id = threadIdx.x >> 5;
        const int lane    = threadIdx.x & 31;

        const int N4 = N >> 2;
        const float4 *x4 = reinterpret_cast<const float4 *>(x);

        for (int r = warp_id; r < row_count; r += WARPS_PER_BLOCK)
        {
            int row = row_begin + r;
            const float4 *A4 = reinterpret_cast<const float4 *>(A + (size_t)row * N);

            float acc = 0.0f;
            #pragma unroll 4
            for (int j = lane; j < N4; j += WARP_SIZE)
            {
                float4 a = A4[j];
                float4 b = x4[j];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            acc = warp_reduce_sum(acc);

            if (lane == 0)
            {
                y[row] += scale * acc;
            }
        }
    }

    template <int THREADS_PER_BLOCK = 128>
    __global__ void gemv_kernel(
        const float *__restrict__ A,
        const float *__restrict__ x,
        float *__restrict__ y,
        int M, int N)
    {
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
        int row_begin = blockIdx.x * WARPS_PER_BLOCK;
        int row_count = min(WARPS_PER_BLOCK, M - row_begin);
        if (row_count <= 0) return;
        gemv_tile<THREADS_PER_BLOCK>(A, x, y, N, row_begin, row_count);
    }

} // namespace flashmoe
