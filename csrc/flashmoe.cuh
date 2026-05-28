#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils/trace.cuh"

enum TraceLabel : int
{
    // Groups (outer spans)
    TR_WAIT,
    TR_FFN1,
    TR_FFN2,
    TR_ROUTE,
    TR_SCHEDULE,
    // Leaves (inner filled bars)
    TR_GEMV_GATE,
    TR_SILU_MUL,
    TR_GEMV_DOWN,
    TR_GEMV_ROUTE,
    TR_SOFTMAX_TOPK,
    TR_DISPATCH,
    TR_NUM_LABELS
};

constexpr int TR_FIRST_LEAF = TR_GEMV_GATE;

inline const char **trace_label_names()
{
    static const char *names[] = {
        "WAIT", "FFN1", "FFN2", "ROUTE", "SCHEDULE",
        "GEMV_GATE", "SILU_MUL", "GEMV_DOWN",
        "GEMV_ROUTE", "SOFTMAX_TOPK", "DISPATCH"};
    return names;
}

namespace constants
{

#ifndef BLOCKSIZE_OVERRIDE
    constexpr int BLOCKSIZE = 46;
#else
    constexpr int BLOCKSIZE = BLOCKSIZE_OVERRIDE;
#endif

#ifndef CAPACITY_OVERRIDE
    constexpr int CAPACITY = 512;
#else
    constexpr int CAPACITY = CAPACITY_OVERRIDE;
#endif

#ifndef HIDDEN_SIZE_OVERRIDE
    constexpr int HIDDEN_SIZE = 2048;
#else
    constexpr int HIDDEN_SIZE = HIDDEN_SIZE_OVERRIDE;
#endif

#ifndef MOE_INTERMEDIATE_SIZE_OVERRIDE
    constexpr int MOE_INTERMEDIATE_SIZE = 768;
#else
    constexpr int MOE_INTERMEDIATE_SIZE = MOE_INTERMEDIATE_SIZE_OVERRIDE;
#endif

#ifndef NUM_EXPERTS_OVERRIDE
    constexpr int NUM_EXPERTS = 128;
#else
    constexpr int NUM_EXPERTS = NUM_EXPERTS_OVERRIDE;
#endif

#ifndef TOP_K_OVERRIDE
    constexpr int TOP_K = 8;
#else
    constexpr int TOP_K = TOP_K_OVERRIDE;
#endif

    constexpr size_t GATE_PROJ_SIZE = HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
    constexpr size_t UP_PROJ_SIZE = HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
    constexpr size_t DOWN_PROJ_SIZE = MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE;
    constexpr size_t ROUTER_SIZE = HIDDEN_SIZE * NUM_EXPERTS;

#ifndef TILE_ROWS_OVERRIDE
    constexpr int TILE_ROWS = 128;
#else
    constexpr int TILE_ROWS = TILE_ROWS_OVERRIDE;
#endif

    constexpr int FFN1_TILES_PER_EXPERT = MOE_INTERMEDIATE_SIZE / TILE_ROWS;
    constexpr int FFN2_TILES_PER_EXPERT = (HIDDEN_SIZE + TILE_ROWS - 1) / TILE_ROWS;

    constexpr int NUM_WORKERS = BLOCKSIZE - 1; // 45
#ifndef TPB_OVERRIDE
    constexpr int THREADS_PER_BLOCK = 128;
#else
    constexpr int THREADS_PER_BLOCK = TPB_OVERRIDE;
#endif
    constexpr int TOTAL_TASKS =
        TOP_K * FFN1_TILES_PER_EXPERT +
        TOP_K * FFN2_TILES_PER_EXPERT;
}

template <typename T>
struct ExpertFFN
{
    T *gate_proj;
    T *up_proj;
    T *down_proj;
};

template <typename T>
struct FlashMoe
{
    ExpertFFN<T> experts[constants::NUM_EXPERTS];
    T *router;
};

template <typename T, typename AccT = __half>
struct MoeState
{
    T *input;         // [H] in storage type
    float *output;    // [H] accumulation target (always fp32)
    AccT *ffn1_out;   // [TOP_K * I] intermediate activations
    int *ffn1_done;   // [TOP_K] fan-in counters
    float *logits;    // [NUM_EXPERTS] router logits (global, all blocks write)
    int *router_done; // atomic counter for router barrier
};
