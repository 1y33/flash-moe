#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct DType;

template <>
struct DType<float>
{
    using store_t = float;
    using compute_t = float;
    static constexpr int VEC = 4;
    using vec_t = float4;

    // vector
    __device__ __forceinline__ static void load_vec(const float *ptr, float out[VEC])
    {
        float4 v = reinterpret_cast<const float4 *>(ptr)[0];
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    }

    __device__ __forceinline__ static void store_vec(float *ptr, const float in[VEC])
    {
        float4 v = {in[0], in[1], in[2], in[3]};
        reinterpret_cast<float4 *>(ptr)[0] = v;
    }

    // scalar
    __device__ __forceinline__ static float load(const float *ptr) { return *ptr; }

    __device__ __forceinline__ static void store(float *ptr, float v) { *ptr = v; }

    // dot of two loaded vectors
    __device__ __forceinline__ static float dot(const float a[VEC], const float b[VEC])
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }
};


template <>
struct DType<__half>
{
    using store_t = __half;
    using compute_t = float;

    static constexpr int VEC = 8;
    using vec_t = uint4;
    
    // vector
    __device__ __forceinline__ static void load_vec(const __half *ptr, float out[VEC])
    {
        uint4 v = reinterpret_cast<const uint4 *>(ptr)[0];
        const __half2 *pairs = reinterpret_cast<const __half2 *>(&v);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            out[2 * i] = __half2float(pairs[i].x);
            out[2 * i + 1] = __half2float(pairs[i].y);
        }
    }

    __device__ __forceinline__ static void store_vec(__half *ptr, const float in[VEC])
    {
        __half2 pairs[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            pairs[i] = __halves2half2(
                __float2half(in[2 * i]),
                __float2half(in[2 * i + 1]));
        }
        reinterpret_cast<uint4 *>(ptr)[0] =
            *reinterpret_cast<uint4 *>(pairs);
    }


    //scalar
    __device__ __forceinline__ static float load(const __half *ptr) { return __half2float(*ptr); }

    __device__ __forceinline__ static void store(__half *ptr, float v) { *ptr = __float2half(v); }

    __device__ __forceinline__ static float dot(const float a[VEC], const float b[VEC])
    {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC; i++)
            acc += a[i] * b[i];
        return acc;
    }
};
