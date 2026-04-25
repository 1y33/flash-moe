#pragma once
#include <cuda_runtime.h>
#include "../utils/warp.cuh"
#include "../utils/dtypes.cuh"

namespace flashmoe
{

    // Prefetched fused gate+up: load next iteration while computing current
    template <typename T, typename AccT, int THREADS_PER_BLOCK = 128, int ILP = 4>
    __device__ __forceinline__ void gemv_tile_fused_gate_up_prefetch(
        const T *__restrict__ A_gate,
        const T *__restrict__ A_up,
        const T *__restrict__ x,
        AccT *__restrict__ y_gate,
        AccT *__restrict__ y_up,
        int N,
        int row_begin,
        int row_count)
    {
        using D = DType<T>;
        using DA = DType<AccT>;
        constexpr int VEC = D::VEC;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / warp::SIZE;

        const int wid = warp::warp_id();
        const int lane = warp::lane_id();
        const int N_VEC = N / VEC;

        for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS_PER_BLOCK * ILP)
        {
            float acc_g[ILP], acc_u[ILP];
#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                acc_g[i] = 0.0f;
                acc_u[i] = 0.0f;
            }

            // Prologue: load first chunk into "cur" registers
            float b_cur[VEC];
            float ag_cur[ILP][VEC], au_cur[ILP][VEC];
            D::load_vec(x + lane * VEC, b_cur);
#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count)
                {
                    size_t off = (size_t)(row_begin + r_base + i) * N + lane * VEC;
                    D::load_vec(A_gate + off, ag_cur[i]);
                    D::load_vec(A_up + off, au_cur[i]);
                }
                else
                {
#pragma unroll
                    for (int k = 0; k < VEC; k++)
                    {
                        ag_cur[i][k] = 0.0f;
                        au_cur[i][k] = 0.0f;
                    }
                }
            }

            // Main loop: prefetch next, compute current
            for (int j = lane + warp::SIZE; j < N_VEC; j += warp::SIZE)
            {
                // Prefetch next chunk
                float b_nxt[VEC];
                float ag_nxt[ILP][VEC], au_nxt[ILP][VEC];
                D::load_vec(x + j * VEC, b_nxt);
#pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    if (r_base + i < row_count)
                    {
                        size_t off = (size_t)(row_begin + r_base + i) * N + j * VEC;
                        D::load_vec(A_gate + off, ag_nxt[i]);
                        D::load_vec(A_up + off, au_nxt[i]);
                    }
                    else
                    {
#pragma unroll
                        for (int k = 0; k < VEC; k++)
                        {
                            ag_nxt[i][k] = 0.0f;
                            au_nxt[i][k] = 0.0f;
                        }
                    }
                }

// Compute on current
#pragma unroll
                for (int i = 0; i < ILP; i++)
                {
                    acc_g[i] += D::dot(ag_cur[i], b_cur);
                    acc_u[i] += D::dot(au_cur[i], b_cur);
                }

// Swap: cur = nxt
#pragma unroll
                for (int k = 0; k < VEC; k++)
                    b_cur[k] = b_nxt[k];
#pragma unroll
                for (int i = 0; i < ILP; i++)
                {
#pragma unroll
                    for (int k = 0; k < VEC; k++)
                    {
                        ag_cur[i][k] = ag_nxt[i][k];
                        au_cur[i][k] = au_nxt[i][k];
                    }
                }
            }

// Epilogue: compute on last loaded chunk
#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                acc_g[i] += D::dot(ag_cur[i], b_cur);
                acc_u[i] += D::dot(au_cur[i], b_cur);
            }

#pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count)
                {
                    acc_g[i] = warp::reduce_sum(acc_g[i]);
                    acc_u[i] = warp::reduce_sum(acc_u[i]);
                    if (lane == 0)
                    {
                        DA::store(&y_gate[row_begin + r_base + i], acc_g[i]);
                        DA::store(&y_up[row_begin + r_base + i], acc_u[i]);
                    }
                }
            }
        }
    }

} // namespace flashmoe
