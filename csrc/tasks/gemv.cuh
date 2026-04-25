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

} // namespace flashmoe
