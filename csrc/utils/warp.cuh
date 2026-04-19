#pragma once
#include <cuda_runtime.h>

namespace warp
{
    constexpr int SIZE = 32;
    constexpr unsigned FULL_MASK = 0xffffffff;

    __device__ __forceinline__ float reduce_sum(float v)
    {
        v += __shfl_down_sync(FULL_MASK, v, 16);
        v += __shfl_down_sync(FULL_MASK, v, 8);
        v += __shfl_down_sync(FULL_MASK, v, 4);
        v += __shfl_down_sync(FULL_MASK, v, 2);
        v += __shfl_down_sync(FULL_MASK, v, 1);
        return v;
    }

    __device__ __forceinline__ float reduce_max(float v)
    {
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 16));
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 8));
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 4));
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 2));
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 1));
        return v;
    }

    __device__ __forceinline__ void argmax(float val, int idx, float &max_val, int &max_idx)
    {
        max_val = val;
        max_idx = idx;

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_val = __shfl_xor_sync(FULL_MASK, max_val, offset);
            int other_idx = __shfl_xor_sync(FULL_MASK, max_idx, offset);
            bool take = (other_val > max_val) ||
                        (other_val == max_val && other_idx < max_idx);
            if (take)
            {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
    }


    __device__ __forceinline__ float broadcast(float v, int src_lane = 0)
    {
        return __shfl_sync(FULL_MASK, v, src_lane);
    }

    __device__ __forceinline__ int broadcast(int v, int src_lane = 0)
    {
        return __shfl_sync(FULL_MASK, v, src_lane);
    }


    __device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
    __device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

} // namespace warp
