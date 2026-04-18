#pragma once
#include <cuda_runtime.h>
#include <cfloat>


namespace flashmoe
{

    __device__ __forceinline__ void warp_argmax(
        float val, int lane, float &max_val, int &max_lane)
    {
        max_val  = val;
        max_lane = lane;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_val  = __shfl_xor_sync(0xffffffff, max_val,  offset);
            int other_lane = __shfl_xor_sync(0xffffffff, max_lane, offset);
            bool take_other = (other_val > max_val) ||
                              (other_val == max_val && other_lane < max_lane);
            if (take_other) { max_val = other_val; max_lane = other_lane; }
        }
    }

    // Device-side primitive. Must be called by exactly 32 threads of one warp.
    //
    // E          : number of experts (compile-time; must be multiple of 32).
    //              Each lane holds E/32 logits, laid out strided:
    //                  lane l owns logits[l], logits[l+32], logits[l+64], ...
    // K          : top-K (compile-time).
    // logits     : [E] in global (or shared) memory.
    // ids_out    : [K] written by lane 0.
    // weights_out: [K] written by lane 0 (softmax-normalized).
    template <int E, int K>
    __device__ __forceinline__ void softmax_topk_warp(
        const float *__restrict__ logits,
        int   *__restrict__ ids_out,
        float *__restrict__ weights_out)
    {
        static_assert(E % 32 == 0, "E must be a multiple of warp size");
        constexpr int PER_LANE = E / 32;

        const int lane = threadIdx.x & 31;

        // Load this lane's logits into registers. Strided layout so that
        // consecutive lanes touch consecutive memory (coalesced load).
        float my[PER_LANE];
        int   my_idx[PER_LANE];
        #pragma unroll
        for (int p = 0; p < PER_LANE; ++p)
        {
            int idx  = p * 32 + lane;
            my[p]     = logits[idx];
            my_idx[p] = idx;
        }

        // Holders for winners (written only by lane 0 at the end).
        int   top_ids[K];
        float top_logits[K];  // keep ORIGINAL logits; we softmax them at the end

        // ---- K rounds of argmax-then-mask ----
        #pragma unroll
        for (int k = 0; k < K; ++k)
        {
            // 1) Each lane finds its local argmax across PER_LANE entries.
            float local_val = -FLT_MAX;
            int   local_idx = -1;
            #pragma unroll
            for (int p = 0; p < PER_LANE; ++p)
            {
                if (my[p] > local_val)
                {
                    local_val = my[p];
                    local_idx = my_idx[p];
                }
            }

            // 2) Warp-wide argmax reduction.
            float max_val;
            int   max_idx;
            warp_argmax(local_val, local_idx, max_val, max_idx);
            // Now every lane has (max_val, max_idx) for this round.

            if (lane == 0)
            {
                top_ids[k]    = max_idx;
                top_logits[k] = max_val;
            }

            // 3) Mask out the winner so it can't win again.
            //    Which lane owned it? lane = max_idx & 31. Which slot? p = max_idx / 32.
            int winner_lane = max_idx & 31;
            int winner_p    = max_idx >> 5;
            if (lane == winner_lane)
            {
                my[winner_p] = -FLT_MAX;
            }
        }

        // ---- Softmax over the K winners (lane 0 only; K is tiny) ----
        if (lane == 0)
        {
            // Numerically-stable softmax: subtract max.
            float m = top_logits[0];
            #pragma unroll
            for (int k = 1; k < K; ++k) m = fmaxf(m, top_logits[k]);

            float sum = 0.0f;
            float e[K];
            #pragma unroll
            for (int k = 0; k < K; ++k)
            {
                e[k] = __expf(top_logits[k] - m);
                sum += e[k];
            }
            float inv = 1.0f / sum;
            #pragma unroll
            for (int k = 0; k < K; ++k)
            {
                ids_out[k]     = top_ids[k];
                weights_out[k] = e[k] * inv;
            }
        }
    }

    // __global__ wrapper for standalone testing.
    template <int E, int K>
    __global__ void softmax_topk_kernel(
        const float *logits, int *ids_out, float *weights_out)
    {
        softmax_topk_warp<E, K>(logits, ids_out, weights_out);
    }

} // namespace flashmoe
