// Benchmark gemv_tile_accumulate_mixed_prefetch (GEMV_DOWN / FFN2 path)
// A = half weights, x = half activations, y = float output (accumulate)
//
// Build:  make test_gemv_down
// Run:    ./build/test_gemv_down

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/utils/dtypes.cuh"
#include "../csrc/tasks/gemv_ffn2.cuh"
#include "bench.cuh"

namespace C = constants;

// ─── Kernel wrapper ─────────────────────────────────────────

template <int TPB, int ILP>
__global__ void kern_down(__half const *A, __half const *x, float *y,
                          int N, int row_begin, int row_count, float scale)
{
    flashmoe::gemv_tile_accumulate_mixed_prefetch<__half, __half, TPB, ILP>(
        A, x, y, N, row_begin, row_count, scale);
}

// ─── CPU reference ──────────────────────────────────────────

static void gemv_ref(const __half *A, const __half *x, float *y,
                     int N, int rb, int rc, float scale)
{
    for (int r = 0; r < rc; r++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += __half2float(A[(rb + r) * N + j]) * __half2float(x[j]);
        y[rb + r] += scale * acc;
    }
}

// ─── Benchmark harness ──────────────────────────────────────

template <typename KernelFn>
static void bench(const char *name, int M, int N, int tile,
                  KernelFn kern, int nruns)
{
    const float scale = 0.125f;

    // Host data
    __half *hA = (__half *)malloc(M * N * sizeof(__half));
    __half *hx = (__half *)malloc(N * sizeof(__half));
    float *hy  = (float *)malloc(M * sizeof(float));
    float *href = (float *)malloc(M * sizeof(float));

    srand(42);
    for (int i = 0; i < M * N; i++)
        hA[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < N; i++)
        hx[i] = __float2half((float)rand() / RAND_MAX - 0.5f);

    memset(href, 0, M * sizeof(float));
    gemv_ref(hA, hx, href, N, 0, tile, scale);

    // Device
    __half *dA, *dx;
    float *dy;
    cudaMalloc(&dA, M * N * sizeof(__half));
    cudaMalloc(&dx, N * sizeof(__half));
    cudaMalloc(&dy, M * sizeof(float));
    cudaMemcpy(dA, hA, M * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, N * sizeof(__half), cudaMemcpyHostToDevice);

    // Warmup
    cudaMemset(dy, 0, M * sizeof(float));
    kern(dA, dx, dy, N, 0, tile, scale);
    cudaDeviceSynchronize();

    // Correctness
    cudaMemset(dy, 0, M * sizeof(float));
    kern(dA, dx, dy, N, 0, tile, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(hy, dy, M * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < tile; i++) {
        float e = fabsf(hy[i] - href[i]);
        if (e > max_err) max_err = e;
    }
    bool pass = max_err < 1e-2f;

    // Timing
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; r++) {
        cudaMemset(dy, 0, M * sizeof(float));
        kern(dA, dx, dy, N, 0, tile, scale);
    }
    t.end();

    // A=half(2B), x=half(2B), y=float(4B)
    double flops = 2.0 * tile * N;
    double bytes = (double)(tile * N) * 2 + (double)N * 2 + (double)tile * 4;
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    printf("  %-36s err=%.1e %s  %.4fms  %.4f TFLOPS  %6.1f GB/s\n",
           name, max_err, pass ? "OK" : "FAIL", res.avg_ms, res.tflops, res.bw_gbps);

    free(hA); free(hx); free(hy); free(href);
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
}

// ─── Main ───────────────────────────────────────────────────

int main()
{
    constexpr int nruns = 500;
    constexpr int M    = C::HIDDEN_SIZE;           // 2048
    constexpr int N    = C::MOE_INTERMEDIATE_SIZE; // 768
    constexpr int tile = C::TILE_ROWS;             // 96

    printf("=== GEMV_DOWN benchmark (half A, half x, float y, accumulate) ===\n");
    printf("Config: TILE=%d  M=%d  N=%d\n\n", tile, M, N);

    printf("── ILP sweep ──\n");
    bench("ILP=4", M, N, tile,
        [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
        { kern_down<128,4><<<1,128>>>(A, x, y, N, rb, rc, s); }, nruns);
    bench("ILP=8", M, N, tile,
        [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
        { kern_down<128,8><<<1,128>>>(A, x, y, N, rb, rc, s); }, nruns);

    printf("\n── TPB sweep ──\n");
    bench("TPB=64  ILP=4", M, N, tile,
        [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
        { kern_down<64,4><<<1,64>>>(A, x, y, N, rb, rc, s); }, nruns);
    bench("TPB=128 ILP=4", M, N, tile,
        [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
        { kern_down<128,4><<<1,128>>>(A, x, y, N, rb, rc, s); }, nruns);
    bench("TPB=256 ILP=4", M, N, tile,
        [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
        { kern_down<256,4><<<1,256>>>(A, x, y, N, rb, rc, s); }, nruns);

    printf("\n── Tile sweep (TPB=128, ILP=4) ──\n");
    int tiles[] = {48, 96, 128, 192, 256, 384, 512, 768, 1024};
    for (int t : tiles) {
        if (t > M) continue;
        char name[64];
        snprintf(name, sizeof(name), "tile=%d", t);
        bench(name, M, N, t,
            [](auto *A, auto *x, auto *y, int N, int rb, int rc, float s)
            { kern_down<128,4><<<1,128>>>(A, x, y, N, rb, rc, s); }, nruns);
    }

    return 0;
}
