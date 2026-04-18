#pragma once
#include <cuda_runtime.h>
#include <cfloat>

// ============================================================================
// Top-K selection of largest logits (no softmax).
// ----------------------------------------------------------------------------
// Method: K rounds of warp-wide argmax, masking the winner after each round.
// Cost  : O(K * (E/32 + log2(32))). For E=128, K=8: ~72 register ops.
//
// One warp (32 threads) must call these. E must be a multiple of 32.
//
// Two flavors:
//   topk_warp_regs     : results live in the caller's *stack arrays* (lane 0
//                        is the only lane whose arrays are meaningful). Used
//                        to compose with softmax or other in-warp pipelines.
//   topk_warp_global   : results are written to pointers in global/shared
//                        memory by lane 0. Used for standalone tests.
// ============================================================================

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
            int   other_lane = __shfl_xor_sync(0xffffffff, max_lane, offset);
            bool take_other = (other_val > max_val) ||
                              (other_val == max_val && other_lane < max_lane);
            if (take_other) { max_val = other_val; max_lane = other_lane; }
        }
    }

    // Compute top-K. On return the caller's `ids[K]` and `top_logits[K]` are
    // valid ONLY on lane 0. This avoids a global-memory round-trip when
    // chaining topk with softmax inside the same warp.
    template <int E, int K>
    __device__ __forceinline__ void topk_warp_regs(
        const float *__restrict__ logits,
        int   (&ids)[K],
        float (&top_logits)[K])
    {
        static_assert(E % 32 == 0, "E must be a multiple of warp size");
        constexpr int PER_LANE = E / 32;

        const int lane = threadIdx.x & 31;

        float my[PER_LANE];
        int   my_idx[PER_LANE];
        #pragma unroll
        for (int p = 0; p < PER_LANE; ++p)
        {
            int idx = p * 32 + lane;
            my[p]     = logits[idx];
            my_idx[p] = idx;
        }

        #pragma unroll
        for (int k = 0; k < K; ++k)
        {
            float local_val = -FLT_MAX;
            int   local_idx = -1;
            #pragma unroll
            for (int p = 0; p < PER_LANE; ++p)
            {
                if (my[p] > local_val) { local_val = my[p]; local_idx = my_idx[p]; }
            }

            float max_val;
            int   max_idx;
            warp_argmax(local_val, local_idx, max_val, max_idx);

            if (lane == 0)
            {
                ids[k]        = max_idx;
                top_logits[k] = max_val;
            }

            int winner_lane = max_idx & 31;
            int winner_p    = max_idx >> 5;
            if (lane == winner_lane) my[winner_p] = -FLT_MAX;
        }
    }

    // Convenience: write results to global/shared memory.
    template <int E, int K>
    __device__ __forceinline__ void topk_warp_global(
        const float *__restrict__ logits,
        int   *__restrict__ ids_out,
        float *__restrict__ top_logits_out)
    {
        int   ids[K];
        float top_logits[K];
        topk_warp_regs<E, K>(logits, ids, top_logits);

        if ((threadIdx.x & 31) == 0)
        {
            #pragma unroll
            for (int k = 0; k < K; ++k)
            {
                ids_out[k]        = ids[k];
                top_logits_out[k] = top_logits[k];
            }
        }
    }

    template <int E, int K>
    __global__ void topk_kernel(
        const float *logits, int *ids_out, float *top_logits_out)
    {
        topk_warp_global<E, K>(logits, ids_out, top_logits_out);
    }

} // namespace flashmoe
