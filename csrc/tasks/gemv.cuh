#pragma once
#include <cuda_runtime.h>
#include "../utils/warp.cuh"
#include "../utils/dtypes.cuh"

namespace flashmoe
{

    template <typename T, int THREADS_PER_BLOCK = 128, int ILP = 4>
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

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc[ILP];
            #pragma unroll
            for (int i = 0; i < ILP; i++) acc[i] = 0.0f;

            #pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                float b[VEC];
                D::load_vec(x + j * VEC, b);

                #pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count) {
                        float a[VEC];
                        D::load_vec(A + (size_t)(row_begin + r_base + i) * N + j * VEC, a);
                        acc[i] += D::dot(a, b);
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    acc[i] = warp::reduce_sum(acc[i]);
                    if (lane == 0)
                        y[row_begin + r_base + i] = acc[i];
                }
            }
        }
    }

    template <typename T, int THREADS_PER_BLOCK = 128, int ILP = 4>
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

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc[ILP];
            #pragma unroll
            for (int i = 0; i < ILP; i++) acc[i] = 0.0f;

            #pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                float b[VEC];
                D::load_vec(x + j * VEC, b);

                #pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count) {
                        float a[VEC];
                        D::load_vec(A + (size_t)(row_begin + r_base + i) * N + j * VEC, a);
                        acc[i] += D::dot(a, b);
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    acc[i] = warp::reduce_sum(acc[i]);
                    if (lane == 0)
                        y[row_begin + r_base + i] += scale * acc[i];
                }
            }
        }
    }

    // Fused gate+up: read x once, compute both projections
    template <typename T, int THREADS_PER_BLOCK = 128, int ILP = 4>
    __device__ __forceinline__ void gemv_tile_fused_gate_up(
        const T *__restrict__ A_gate,
        const T *__restrict__ A_up,
        const T *__restrict__ x,
        float   *__restrict__ y_gate,
        float   *__restrict__ y_up,
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

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc_g[ILP], acc_u[ILP];
            #pragma unroll
            for (int i = 0; i < ILP; i++) { acc_g[i] = 0.0f; acc_u[i] = 0.0f; }

            #pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                float b[VEC];
                D::load_vec(x + j * VEC, b);

                #pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count) {
                        size_t off = (size_t)(row_begin + r_base + i) * N + j * VEC;
                        float ag[VEC], au[VEC];
                        D::load_vec(A_gate + off, ag);
                        D::load_vec(A_up   + off, au);
                        acc_g[i] += D::dot(ag, b);
                        acc_u[i] += D::dot(au, b);
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    acc_g[i] = warp::reduce_sum(acc_g[i]);
                    acc_u[i] = warp::reduce_sum(acc_u[i]);
                    if (lane == 0) {
                        y_gate[row_begin + r_base + i] = acc_g[i];
                        y_up[row_begin + r_base + i]   = acc_u[i];
                    }
                }
            }
        }
    }

    // Mixed-precision accumulate: A is type T (e.g. half), x is float
    // Iterates using DType<T>::VEC for A loads, and manually loads matching floats from x
    template <typename T, int THREADS_PER_BLOCK = 128, int ILP = 4>
    __device__ __forceinline__ void gemv_tile_accumulate_mixed(
        const T     *__restrict__ A,
        const float *__restrict__ x,
        float       *__restrict__ y,
        int N,
        int row_begin,
        int row_count,
        float scale)
    {
        using D = DType<T>;
        constexpr int VEC = D::VEC;  // 8 for half, 4 for float
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;

        const int wid  = warp::warp_id();
        const int lane = warp::lane_id();
        const int N_VEC = N / VEC;

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc[ILP];
            #pragma unroll
            for (int i = 0; i < ILP; i++) acc[i] = 0.0f;

            #pragma unroll 4
            for (int j = lane; j < N_VEC; j += warp::SIZE)
            {
                // Load VEC floats from x
                float b[VEC];
                #pragma unroll
                for (int k = 0; k < VEC; k++)
                    b[k] = x[j * VEC + k];

                #pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count) {
                        float a[VEC];
                        D::load_vec(A + (size_t)(row_begin + r_base + i) * N + j * VEC, a);
                        acc[i] += D::dot(a, b);
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    acc[i] = warp::reduce_sum(acc[i]);
                    if (lane == 0)
                        y[row_begin + r_base + i] += scale * acc[i];
                }
            }
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
        gemv_tile<T, THREADS_PER_BLOCK, 1>(A, x, y, N, row_begin, row_count);
    }

} // namespace flashmoe
