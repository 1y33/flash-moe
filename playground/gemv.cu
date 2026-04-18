#include <cstdio>
#include <cmath>
#include "../csrc/allocator.cu"
#include "bench.cuh"

// v1: naive — 1 thread per row
template <int BLOCK_SIZE = 256>
__global__ void gemv_1(
    const float *A,
    const float *B,
    float *C, int M, int N)
{
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= M) return;

    float acc = 0.0f;
    for (int i = 0; i < N; i++) {
        acc += A[row * N + i] * B[i];
    }
    C[row] = acc;
}

// v2: warp shuffle — 1 warp (32 threads) per row, float4 vectorized loads
template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void gemv_2(
    const float *A,
    const float *B,
    float *C, int M, int N)
{
    int row  = (blockIdx.x * BLOCK_SIZE + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= M) return;

    const float4 *A4 = reinterpret_cast<const float4 *>(A + row * N);
    const float4 *B4 = reinterpret_cast<const float4 *>(B);
    int N4 = N / 4;

    float acc = 0.0f;
    for (int j = lane; j < N4; j += WARP_SIZE) {
        float4 a = A4[j];
        float4 b = B4[j];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    acc += __shfl_down_sync(0xffffffff, acc, 16);
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 4);
    acc += __shfl_down_sync(0xffffffff, acc, 2);
    acc += __shfl_down_sync(0xffffffff, acc, 1);

    if (lane == 0) C[row] = acc;
}

// v3: shared memory tiled + float4 vectorized loads
// Tile size = BLOCK_SIZE floats. 256 threads each load 1 float per tile.
template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void gemv_3(
    const float *A,
    const float *B,
    float *C, int M, int N)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= M) return;

    extern __shared__ float smem_B[];

    float acc = 0.0f;

    // Each tile: BLOCK_SIZE elements of B loaded into smem
    for (int tile_start = 0; tile_start < N; tile_start += BLOCK_SIZE) {
        // All 256 threads load 1 element each → 256 elements in smem
        int idx = tile_start + threadIdx.x;
        smem_B[threadIdx.x] = (idx < N) ? B[idx] : 0.0f;
        __syncthreads();

        // Each warp reads its row's A tile with float4, multiplies with smem_B
        int tile_elems = min(BLOCK_SIZE, N - tile_start);
        int elems4 = tile_elems / 4;
        const float4 *A4 = reinterpret_cast<const float4 *>(A + row * N + tile_start);
        const float4 *S4 = reinterpret_cast<const float4 *>(smem_B);
        for (int j = lane; j < elems4; j += WARP_SIZE) {
            float4 a = A4[j];
            float4 s = S4[j];
            acc += a.x * s.x + a.y * s.y + a.z * s.z + a.w * s.w;
        }
        __syncthreads();
    }

    // Warp shuffle reduce
    acc += __shfl_down_sync(0xffffffff, acc, 16);
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 4);
    acc += __shfl_down_sync(0xffffffff, acc, 2);
    acc += __shfl_down_sync(0xffffffff, acc, 1);

    if (lane == 0) C[row] = acc;
}

void gemv_cpu(const float *A, const float *B, float *C, int M, int N) {
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += A[i * N + j] * B[j];
        }
        C[i] = acc;
    }
}

// Autotune macro for warp-based kernels (1 warp per row)
// USE_SMEM: 0 = no smem, 1 = BS*sizeof(float) bytes
#define TRY_WARP_KERNEL(KERNEL, BS, USE_SMEM) {                               \
    constexpr int _WARPS = BS / 32;                                            \
    dim3 _block(BS);                                                           \
    dim3 _grid((M + _WARPS - 1) / _WARPS);                                    \
    int _smem = USE_SMEM ? (BS * (int)sizeof(float)) : 0;                     \
    KERNEL<BS><<<_grid, _block, _smem>>>(d_A, d_B, d_C, M, N);               \
    cudaDeviceSynchronize();                                                    \
    GpuTimer _t; _t.begin();                                                   \
    for (int _r = 0; _r < nruns; _r++)                                        \
        KERNEL<BS><<<_grid, _block, _smem>>>(d_A, d_B, d_C, M, N);           \
    _t.end();                                                                   \
    CudaAllocator::copy_to_host(d_C, h_C, M);                                 \
    BenchResult _res = Bench::gemv(M, N, nruns, _t);                           \
    bool _pass = Bench::verify(h_C, h_ref, M);                                \
    printf("  " #KERNEL "<BS=%3d>  %s", BS, _pass ? "" : "*** FAIL *** ");    \
    _res.print();                                                               \
    if (_res.avg_ms < best.avg_ms) { best = _res; best_bs = BS; }             \
}

// Autotune macro for naive kernel (1 thread per row)
#define TRY_NAIVE_KERNEL(KERNEL, BS) {                                         \
    dim3 _block(BS);                                                           \
    dim3 _grid((M + BS - 1) / BS);                                            \
    KERNEL<BS><<<_grid, _block>>>(d_A, d_B, d_C, M, N);                      \
    cudaDeviceSynchronize();                                                    \
    GpuTimer _t; _t.begin();                                                   \
    for (int _r = 0; _r < nruns; _r++)                                        \
        KERNEL<BS><<<_grid, _block>>>(d_A, d_B, d_C, M, N);                  \
    _t.end();                                                                   \
    CudaAllocator::copy_to_host(d_C, h_C, M);                                 \
    BenchResult _res = Bench::gemv(M, N, nruns, _t);                           \
    bool _pass = Bench::verify(h_C, h_ref, M);                                \
    printf("  " #KERNEL "<BS=%3d>  %s", BS, _pass ? "" : "*** FAIL *** ");    \
    _res.print();                                                               \
    if (_res.avg_ms < best.avg_ms) { best = _res; best_bs = BS; }             \
}

#define SWEEP_NAIVE(KERNEL) {                                                  \
    printf("\n=== Autotune " #KERNEL "  M=%d N=%d ===\n", M, N);              \
    BenchResult best = {999.0f, 0, 0}; int best_bs = 0;                       \
    TRY_NAIVE_KERNEL(KERNEL, 32);                                              \
    TRY_NAIVE_KERNEL(KERNEL, 64);                                              \
    TRY_NAIVE_KERNEL(KERNEL, 128);                                             \
    TRY_NAIVE_KERNEL(KERNEL, 256);                                             \
    TRY_NAIVE_KERNEL(KERNEL, 512);                                             \
    TRY_NAIVE_KERNEL(KERNEL, 1024);                                            \
    printf("  >> BEST: BS=%d  time=%.4f ms  TFLOPS=%.4f  BW=%.2f GB/s\n\n",   \
           best_bs, best.avg_ms, best.tflops, best.bw_gbps);                  \
}

#define SWEEP_WARP(KERNEL, SMEM_EXPR) {                                        \
    printf("\n=== Autotune " #KERNEL "  M=%d N=%d ===\n", M, N);              \
    BenchResult best = {999.0f, 0, 0}; int best_bs = 0;                       \
    TRY_WARP_KERNEL(KERNEL, 32, SMEM_EXPR);                                   \
    TRY_WARP_KERNEL(KERNEL, 64, SMEM_EXPR);                                   \
    TRY_WARP_KERNEL(KERNEL, 128, SMEM_EXPR);                                  \
    TRY_WARP_KERNEL(KERNEL, 256, SMEM_EXPR);                                  \
    TRY_WARP_KERNEL(KERNEL, 512, SMEM_EXPR);                                  \
    TRY_WARP_KERNEL(KERNEL, 1024, SMEM_EXPR);                                 \
    printf("  >> BEST: BS=%d  time=%.4f ms  TFLOPS=%.4f  BW=%.2f GB/s\n\n",   \
           best_bs, best.avg_ms, best.tflops, best.bw_gbps);                  \
}

int main()
{
    int M = 768;
    int N = 2048;
    int nruns = 100;

    float *h_A, *h_B, *h_C, *h_ref;
    HostAllocator::allocate(&h_A, M * N);
    HostAllocator::allocate(&h_B, N);
    HostAllocator::allocate(&h_C, M);
    HostAllocator::allocate(&h_ref, M);

    srand(42);
    HostAllocator::fill_random(h_A, M * N);
    HostAllocator::fill_random(h_B, N);

    gemv_cpu(h_A, h_B, h_ref, M, N);

    float *d_A, *d_B, *d_C;
    CudaAllocator::allocate(&d_A, M * N);
    CudaAllocator::allocate(&d_B, N);
    CudaAllocator::allocate(&d_C, M);

    CudaAllocator::copy_to_device(h_A, d_A, M * N);
    CudaAllocator::copy_to_device(h_B, d_B, N);

    SWEEP_NAIVE(gemv_1);
    SWEEP_WARP(gemv_2, 0);
    SWEEP_WARP(gemv_3, 1);

    HostAllocator::free(h_A); HostAllocator::free(h_B);
    HostAllocator::free(h_C); HostAllocator::free(h_ref);
    CudaAllocator::free(d_A); CudaAllocator::free(d_B); CudaAllocator::free(d_C);

    return 0;
}
