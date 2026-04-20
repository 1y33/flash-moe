#pragma once
#include <cuda_runtime.h>
#include "utils/trace.cuh"

// Trace labels for the persistent kernel
// Group labels (rendered as bordered spans containing leaf labels)
enum TraceLabel : int {
    // Groups (outer spans)
    TR_WAIT,
    TR_FFN1,
    TR_FFN2,
    TR_ROUTE,
    TR_SCHEDULE,
    // Leaves (inner filled bars)
    TR_GEMV_GATE,
    TR_GEMV_UP,
    TR_SILU_MUL,
    TR_GEMV_DOWN,
    TR_GEMV_ROUTE,
    TR_SOFTMAX_TOPK,
    TR_DISPATCH,
    TR_NUM_LABELS
};

// Labels with index < TR_FIRST_LEAF are group spans
constexpr int TR_FIRST_LEAF = TR_GEMV_GATE;

inline const char** trace_label_names() {
    static const char* names[] = {
        "WAIT", "FFN1", "FFN2", "ROUTE", "SCHEDULE",
        "GEMV_GATE", "GEMV_UP", "SILU_MUL", "GEMV_DOWN",
        "GEMV_ROUTE", "SOFTMAX_TOPK", "DISPATCH"
    };
    return names;
}

namespace constants
{

    constexpr int BLOCKSIZE = 46;
    constexpr int CAPACITY = 512;
    constexpr int HIDDEN_SIZE = 2048;

    constexpr int MOE_INTERMEDIATE_SIZE = 768;
    constexpr int NUM_EXPERTS = 128;
    constexpr int TOP_K = 8;

    constexpr size_t GATE_PROJ_SIZE = HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
    constexpr size_t UP_PROJ_SIZE = HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
    constexpr size_t DOWN_PROJ_SIZE = MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE;
    constexpr size_t ROUTER_SIZE = HIDDEN_SIZE * NUM_EXPERTS;

    constexpr int TILE_ROWS = 96;
    constexpr int FFN1_TILES_PER_EXPERT = MOE_INTERMEDIATE_SIZE / TILE_ROWS; // 48
    constexpr int FFN2_TILES_PER_EXPERT = (HIDDEN_SIZE + TILE_ROWS - 1) / TILE_ROWS; // 128

    constexpr int NUM_WORKERS = BLOCKSIZE - 1; // 45
    constexpr int THREADS_PER_BLOCK = 128;
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

template <typename T>
struct MoeState
{
    T     *input;       // [H] in storage type
    float *output;      // [H] final result (always fp32, accumulate target)
    float *ffn1_out;    // [TOP_K * I] intermediate activations (always fp32)
    int   *ffn1_done;   // [TOP_K] fan-in counters
};
