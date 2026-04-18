#pragma once
#include <cuda_runtime.h>
#include "gemv.cuh"
#include "silu_mul.cuh"

// ============================================================================
// Expert FFN building blocks.
// ----------------------------------------------------------------------------
// An MoE expert is SwiGLU:
//
//     FFN1(x) = silu(x @ gate_proj^T) * (x @ up_proj^T)     -> [I]
//     FFN2(x) = FFN1(x) @ down_proj^T                        -> [H]
//
//     expert_out = FFN2(FFN1(x))
//
// With weight layouts:
//     gate_proj : [I, H]   (row-major: row i = the i-th output neuron)
//     up_proj   : [I, H]
//     down_proj : [H, I]
//
// These composites are DEVICE FUNCTIONS: a persistent worker SM calls them
// from within its task loop. Each function expects ONE threadblock and
// partitions work across the block's warps.
//
// Shared-memory layout when calling FFN1:
//   - gate_buf[I] and up_buf[I] are scratch written by FFN1 and consumed
//     by silu_mul.  Caller passes them via shared memory (or HBM scratch
//     for the standalone test).
//
// Numerics: everything is float32 at this stage. A fp16/bf16 version with
// a float32 accumulator is a straightforward extension.
// ============================================================================

namespace flashmoe
{

    // FFN1: writes silu(gate) * up into `act_out[I]`.
    //
    // gate_buf / up_buf are I-sized scratch (shared memory works too).
    // x is [H]. gate_w and up_w are [I, H] row-major.
    template <int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void ffn1_tile(
        const float *__restrict__ x,        // [H]
        const float *__restrict__ gate_w,   // [I, H]
        const float *__restrict__ up_w,     // [I, H]
        float       *__restrict__ gate_buf, // [I] scratch
        float       *__restrict__ up_buf,   // [I] scratch
        float       *__restrict__ act_out,  // [I]
        int H, int I)
    {
        // Two independent GEMVs. The block's warps handle all I rows of each.
        gemv_tile<THREADS_PER_BLOCK>(gate_w, x, gate_buf, /*N=*/H,
                                     /*row_begin=*/0, /*row_count=*/I);
        gemv_tile<THREADS_PER_BLOCK>(up_w,   x, up_buf,   /*N=*/H,
                                     /*row_begin=*/0, /*row_count=*/I);
        __syncthreads();   // ensure both GEMVs are visible before silu_mul

        silu_mul_tile<THREADS_PER_BLOCK>(gate_buf, up_buf, act_out, I);
    }

    // FFN2: y[H] += scale * (down_w @ act)
    //
    // act is [I]. down_w is [H, I] row-major. scale is the router weight
    // for this expert.
    template <int THREADS_PER_BLOCK = 128>
    __device__ __forceinline__ void ffn2_tile_accumulate(
        const float *__restrict__ act,    // [I]
        const float *__restrict__ down_w, // [H, I]
        float       *__restrict__ y,      // [H]
        int H, int I,
        float scale)
    {
        gemv_tile_accumulate<THREADS_PER_BLOCK>(
            down_w, act, y, /*N=*/I,
            /*row_begin=*/0, /*row_count=*/H, scale);
    }

    // ------------------------------------------------------------------------
    // Standalone __global__ wrappers for tests.
    // ------------------------------------------------------------------------
    template <int THREADS_PER_BLOCK = 128>
    __global__ void ffn1_kernel(
        const float *x,
        const float *gate_w,
        const float *up_w,
        float       *gate_buf,
        float       *up_buf,
        float       *act_out,
        int H, int I)
    {
        ffn1_tile<THREADS_PER_BLOCK>(x, gate_w, up_w, gate_buf, up_buf, act_out, H, I);
    }

    template <int THREADS_PER_BLOCK = 128>
    __global__ void ffn2_kernel(
        const float *act,
        const float *down_w,
        float       *y,
        int H, int I,
        float scale)
    {
        ffn2_tile_accumulate<THREADS_PER_BLOCK>(act, down_w, y, H, I, scale);
    }

} // namespace flashmoe
