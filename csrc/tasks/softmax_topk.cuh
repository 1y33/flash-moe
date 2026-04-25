#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include "../utils/warp.cuh"

namespace flashmoe
{

    template <int E, int K>
    __device__ __forceinline__ void softmax_topk_warp(
        const float *__restrict__ logits,
        int *__restrict__ ids_out,
        float *__restrict__ weights_out)
    {
        static_assert(E % 32 == 0, "E must be a multiple of warp size");
        constexpr int PER_LANE = E / 32;

        const int lane = warp::lane_id();

        float my[PER_LANE];
        int my_idx[PER_LANE];
        #pragma unroll
        for (int p = 0; p < PER_LANE; ++p)
        {
            int idx = p * 32 + lane;
            my[p] = logits[idx];
            my_idx[p] = idx;
        }

        int top_ids[K];
        float top_logits[K];

        #pragma unroll
        for (int k = 0; k < K; ++k)
        {
            float local_val = -FLT_MAX;
            int local_idx = -1;
            #pragma unroll
            for (int p = 0; p < PER_LANE; ++p)
            {
                if (my[p] > local_val)
                {
                    local_val = my[p];
                    local_idx = my_idx[p];
                }
            }

            float max_val;
            int max_idx;
            warp::argmax(local_val, local_idx, max_val, max_idx);

            if (lane == 0)
            {
                top_ids[k] = max_idx;
                top_logits[k] = max_val;
            }

            int winner_lane = max_idx & 31;
            int winner_p = max_idx >> 5;
            if (lane == winner_lane)
                my[winner_p] = -FLT_MAX;
        }

        if (lane == 0)
        {
            float m = top_logits[0];
            #pragma unroll
            for (int k = 1; k < K; ++k)
                m = fmaxf(m, top_logits[k]);

            float sum = 0.0f;
            float e[K];
            #pragma unroll
            for (int k = 0; k < K; ++k)
            {
                e[k] = __expf(top_logits[k] - m);
                sum += e[k];
            }
            float inv = __frcp_rn(sum);
            #pragma unroll
            for (int k = 0; k < K; ++k)
            {
                ids_out[k] = top_ids[k];
                weights_out[k] = e[k] * inv;
            }
        }
    }

    template <int E, int K>
    __global__ void softmax_topk_kernel(
        const float *logits, int *ids_out, float *weights_out)
    {
        softmax_topk_warp<E, K>(logits, ids_out, weights_out);
    }

} // namespace flashmoe
