#pragma once
#include <cuda_runtime.h>

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
    constexpr int FFN1_TILES_PER_EXPERT = MOE_INTERMEDIATE_SIZE / TILE_ROWS; // 8
    constexpr int FFN2_TILES_PER_EXPERT = (HIDDEN_SIZE + TILE_ROWS - 1) / TILE_ROWS; // 22

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
