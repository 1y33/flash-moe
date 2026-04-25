#pragma once
#include <cuda_runtime.h>
#include "../utils/warp.cuh"
#include "../utils/dtypes.cuh"

namespace flashmoe
{

    // Prefetched mixed-precision accumulate: load next iteration while computing current
    template <typename T, typename AccT, int THREADS_PER_BLOCK = 128, int ILP = 4>
    __device__ __forceinline__ void gemv_tile_accumulate_mixed_prefetch(
        const T *__restrict__ A,
        const AccT *__restrict__ x,
        float *__restrict__ y,
        int N,
        int row_begin,
        int row_count,
        float scale)
    {
        using D = DType<T>;
        using DA = DType<AccT>;
        constexpr int VEC = D::VEC;
        static_assert(DA::VEC == VEC, "AccT and T must have same VEC for vectorized loads");
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;

        const int wid = warp::warp_id();
        const int lane = warp::lane_id();
        const int N_VEC = N / VEC;

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc[ILP];
#pragma unroll
            for (int i = 0; i < ILP; i++)
                acc[i] = 0.0f;

            // Prologue: load first chunk
            float b_cur[VEC];
            float a_cur[ILP][VEC];
            DA::load_vec(x + lane * VEC, b_cur);
#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count)
                {
                    D::load_vec(A + (size_t)(row_begin + r_base + i) * N + lane * VEC, a_cur[i]);
                }
                else
                {
#pragma unroll
                    for (int k = 0; k < VEC; k++)
                        a_cur[i][k] = 0.0f;
                }
            }

            // Main loop: prefetch next, compute current
            for (int j = lane + warp::SIZE; j < N_VEC; j += warp::SIZE)
            {
                float b_nxt[VEC];
                float a_nxt[ILP][VEC];
                DA::load_vec(x + j * VEC, b_nxt);
#pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count)
                    {
                        D::load_vec(A + (size_t)(row_begin + r_base + i) * N + j * VEC, a_nxt[i]);
                    }
                    else
                    {
#pragma unroll
                        for (int k = 0; k < VEC; k++)
                            a_nxt[i][k] = 0.0f;
                    }
                }

#pragma unroll
                for (int i = 0; i < ILP; i++)
                    acc[i] += D::dot(a_cur[i], b_cur);

#pragma unroll
                for (int k = 0; k < VEC; k++)
                    b_cur[k] = b_nxt[k];
#pragma unroll
                for (int i = 0; i < ILP; i++)
                {
#pragma unroll
                    for (int k = 0; k < VEC; k++)
                        a_cur[i][k] = a_nxt[i][k];
                }
            }

// Epilogue
#pragma unroll
            for (int i = 0; i < ILP; i++)
                acc[i] += D::dot(a_cur[i], b_cur);

#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count)
                {
                    acc[i] = warp::reduce_sum(acc[i]);
                    if (lane == 0)
                        y[row_begin + r_base + i] += scale * acc[i];
                }
            }
        }
    }

} // namespace flashmoe
