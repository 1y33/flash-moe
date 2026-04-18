#pragma once
#include <cuda_runtime.h>


namespace flashmoe
{

    __device__ __forceinline__ float silu(float x)
    {
        return x / (1.0f + __expf(-x));
    }

    template <int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void silu_mul_tile(
        const float *__restrict__ gate,
        const float *__restrict__ up,
        float       *__restrict__ out,
        int I)
    {
        const int I4 = I >> 2;
        const float4 *g4 = reinterpret_cast<const float4 *>(gate);
        const float4 *u4 = reinterpret_cast<const float4 *>(up);
        float4 *o4 = reinterpret_cast<float4 *>(out);

        for (int j = threadIdx.x; j < I4; j += THREADS_PER_BLOCK)
        {
            float4 g = g4[j];
            float4 u = u4[j];
            float4 r;
            r.x = silu(g.x) * u.x;
            r.y = silu(g.y) * u.y;
            r.z = silu(g.z) * u.z;
            r.w = silu(g.w) * u.w;
            o4[j] = r;
        }
    }

    template <int THREADS_PER_BLOCK = 128>
    __global__ void silu_mul_kernel(
        const float *gate, const float *up, float *out, int I)
    {
        silu_mul_tile<THREADS_PER_BLOCK>(gate, up, out, I);
    }

} // namespace flashmoe
