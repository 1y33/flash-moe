#pragma once
#include <cuda_runtime.h>
#include "../utils/warp.cuh"
#include "../utils/dtypes.cuh"

namespace flashmoe
{

    template <typename T, int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void gemv_tile(
        const T *__restrict__ A,
        const T *__restrict__ x,
        float   *__restrict__ y,
        int N,
        int row_begin,
        int row_count)
    {
        using D = DType<T>;
        constexpr int VEC = D::VEC;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;

        const int wid  = warp::warp_id();
        const int lane = warp::lane_id();
        const int N_VEC = N / VEC;

        for (int r = wid; r < row_count; r += WARPS_PER_BLOCK)
        {
            int row = row_begin + r;
            float acc = 0.0f;

#pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                float a[VEC], b[VEC];
                D::load_vec(A + (size_t)row * N + j * VEC, a);
                D::load_vec(x + j * VEC, b);
                acc += D::dot(a, b);
            }

            acc = warp::reduce_sum(acc);
            if (lane == 0)
                y[row] = acc;
        }
    }

    template <typename T, int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void gemv_tile_accumulate(
        const T *__restrict__ A,
        const T *__restrict__ x,
        float   *__restrict__ y,
        int N,
        int row_begin,
        int row_count,
        float scale)
    {
        using D = DType<T>;
        constexpr int VEC = D::VEC;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;

        const int wid  = warp::warp_id();
        const int lane = warp::lane_id();
        const int N_VEC = N / VEC;

        for (int r = wid; r < row_count; r += WARPS_PER_BLOCK)
        {
            int row = row_begin + r;
            float acc = 0.0f;

#pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                float a[VEC], b[VEC];
                D::load_vec(A + (size_t)row * N + j * VEC, a);
                D::load_vec(x + j * VEC, b);
                acc += D::dot(a, b);
            }

            acc = warp::reduce_sum(acc);
            if (lane == 0)
                y[row] += scale * acc;
        }
    }

    template <typename T, int THREADS_PER_BLOCK = 128>
    __global__ void gemv_kernel(
        const T *__restrict__ A,
        const T *__restrict__ x,
        float   *__restrict__ y,
        int M, int N)
    {
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;
        int row_begin = blockIdx.x * WARPS_PER_BLOCK;
        int row_count = min(WARPS_PER_BLOCK, M - row_begin);
        if (row_count <= 0)
            return;
        gemv_tile<T, THREADS_PER_BLOCK>(A, x, y, N, row_begin, row_count);
    }

} // namespace flashmoe
