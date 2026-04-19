#pragma once
#include <cuda_runtime.h>
#include "../utils/dtypes.cuh"

namespace flashmoe
{

    __device__ __forceinline__ float silu(float x)
    {
        return x / (1.0f + __expf(-x));
    }

    template <typename T, int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void silu_mul_tile(
        const T *__restrict__ gate,
        const T *__restrict__ up,
        T       *__restrict__ out,
        int I)
    {
        using D = DType<T>;
        constexpr int VEC = D::VEC;
        const int I_VEC = I / VEC;

        for (int j = threadIdx.x; j < I_VEC; j += THREADS_PER_BLOCK)
        {
            float g[VEC], u[VEC], r[VEC];
            D::load_vec(gate + j * VEC, g);
            D::load_vec(up   + j * VEC, u);

            #pragma unroll
            for (int i = 0; i < VEC; i++)
                r[i] = silu(g[i]) * u[i];

            D::store_vec(out + j * VEC, r);
        }
    }

    template <typename T, int THREADS_PER_BLOCK = 128>
    __global__ void silu_mul_kernel(
        const T *gate, const T *up, T *out, int I)
    {
        silu_mul_tile<T, THREADS_PER_BLOCK>(gate, up, out, I);
    }

} // namespace flashmoe
