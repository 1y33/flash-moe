// Benchmark the production GEMV kernels at persistent kernel tile sizes.
// Single block, TILE_ROWS rows — simulates one worker task.
//
// Tests: gemv_tile (router), gemv_tile_fused_gate_up_prefetch (FFN1)
//
// Build:  make test_gemv_tile
// Run:    ./build/test_gemv_tile

#include <cstdio>
#include <cmath>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/gemv.cuh"
#include "../csrc/tasks/gemv_ffn1.cuh"
#include "bench.cuh"

namespace C = constants;

// ─── Kernel wrappers ────────────────────────────────────────

template <int TPB, int ILP>
__global__ void kern_gemv(const float *A, const float *x, float *y,
                          int N, int rb, int rc) {
    flashmoe::gemv_tile<float, TPB, ILP>(A, x, y, N, rb, rc);
}

template <int TPB, int ILP>
__global__ void kern_fused_prefetch(const float *Ag, const float *Au, const float *x,
                                    float *yg, float *yu, int N, int rb, int rc) {
    flashmoe::gemv_tile_fused_gate_up_prefetch<float, float, TPB, ILP>(Ag, Au, x, yg, yu, N, rb, rc);
}

// ─── CPU reference ──────────────────────────────────────────

static void gemv_ref(const float *A, const float *x, float *y,
                     int N, int rb, int rc) {
    for (int r = 0; r < rc; r++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += A[(rb + r) * N + j] * x[j];
        y[rb + r] = acc;
    }
}

// ─── Harness: single GEMV ───────────────────────────────────

template <typename KernelFn>
static void bench_single(const char *name, int M, int N, int tile,
                          KernelFn kern, int nruns)
{
    float *hA, *hx, *hy, *href;
    HostAllocator::allocate(&hA, M * N);
    HostAllocator::allocate(&hx, N);
    HostAllocator::allocate(&hy, M);
    HostAllocator::allocate(&href, M);

    srand(42);
    HostAllocator::fill_random(hA, M * N);
    HostAllocator::fill_random(hx, N);
    memset(href, 0, M * sizeof(float));
    gemv_ref(hA, hx, href, N, 0, tile);

    float *dA, *dx, *dy;
    CudaAllocator::allocate(&dA, M * N);
    CudaAllocator::allocate(&dx, N);
    CudaAllocator::allocate(&dy, M);
    CudaAllocator::copy_to_device(hA, dA, M * N);
    CudaAllocator::copy_to_device(hx, dx, N);

    cudaMemset(dy, 0, M * sizeof(float));
    kern(dA, dx, dy, N, 0, tile);
    cudaDeviceSynchronize();

    cudaMemset(dy, 0, M * sizeof(float));
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; r++)
        kern(dA, dx, dy, N, 0, tile);
    t.end();
    CudaAllocator::copy_to_host(dy, hy, M);

    float max_err = 0;
    for (int i = 0; i < tile; i++) {
        float e = fabsf(hy[i] - href[i]);
        if (e > max_err) max_err = e;
    }

    double flops = 2.0 * tile * N;
    double bytes = (double)(tile * N + N + tile) * sizeof(float);
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    printf("  %-32s err=%.1e %s  %.4fms  %.4f TFLOPS  %6.1f GB/s\n",
           name, max_err, max_err < 1e-3f ? "OK" : "FAIL",
           res.avg_ms, res.tflops, res.bw_gbps);

    HostAllocator::free(hA); HostAllocator::free(hx);
    HostAllocator::free(hy); HostAllocator::free(href);
    CudaAllocator::free(dA); CudaAllocator::free(dx); CudaAllocator::free(dy);
}

// ─── Harness: fused gate+up ─────────────────────────────────

template <typename KernelFn>
static void bench_fused(const char *name, int M, int N, int tile,
                         KernelFn kern, int nruns)
{
    float *hAg, *hAu, *hx, *hyg, *hyu, *hrg, *hru;
    HostAllocator::allocate(&hAg, M * N);
    HostAllocator::allocate(&hAu, M * N);
    HostAllocator::allocate(&hx, N);
    HostAllocator::allocate(&hyg, M);
    HostAllocator::allocate(&hyu, M);
    HostAllocator::allocate(&hrg, M);
    HostAllocator::allocate(&hru, M);

    srand(42);
    HostAllocator::fill_random(hAg, M * N);
    HostAllocator::fill_random(hAu, M * N);
    HostAllocator::fill_random(hx, N);
    memset(hrg, 0, M * sizeof(float));
    memset(hru, 0, M * sizeof(float));
    gemv_ref(hAg, hx, hrg, N, 0, tile);
    gemv_ref(hAu, hx, hru, N, 0, tile);

    float *dAg, *dAu, *dx, *dyg, *dyu;
    CudaAllocator::allocate(&dAg, M * N);
    CudaAllocator::allocate(&dAu, M * N);
    CudaAllocator::allocate(&dx, N);
    CudaAllocator::allocate(&dyg, M);
    CudaAllocator::allocate(&dyu, M);
    CudaAllocator::copy_to_device(hAg, dAg, M * N);
    CudaAllocator::copy_to_device(hAu, dAu, M * N);
    CudaAllocator::copy_to_device(hx, dx, N);

    cudaMemset(dyg, 0, M * sizeof(float));
    cudaMemset(dyu, 0, M * sizeof(float));
    kern(dAg, dAu, dx, dyg, dyu, N, 0, tile);
    cudaDeviceSynchronize();

    cudaMemset(dyg, 0, M * sizeof(float));
    cudaMemset(dyu, 0, M * sizeof(float));
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; r++)
        kern(dAg, dAu, dx, dyg, dyu, N, 0, tile);
    t.end();
    CudaAllocator::copy_to_host(dyg, hyg, M);
    CudaAllocator::copy_to_host(dyu, hyu, M);

    float max_err = 0;
    for (int i = 0; i < tile; i++) {
        float eg = fabsf(hyg[i] - hrg[i]);
        float eu = fabsf(hyu[i] - hru[i]);
        if (eg > max_err) max_err = eg;
        if (eu > max_err) max_err = eu;
    }

    double flops = 2.0 * 2 * tile * N;
    double bytes = (double)(2 * tile * N + N + 2 * tile) * sizeof(float);
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    printf("  %-32s err=%.1e %s  %.4fms  %.4f TFLOPS  %6.1f GB/s\n",
           name, max_err, max_err < 1e-3f ? "OK" : "FAIL",
           res.avg_ms, res.tflops, res.bw_gbps);

    HostAllocator::free(hAg); HostAllocator::free(hAu); HostAllocator::free(hx);
    HostAllocator::free(hyg); HostAllocator::free(hyu);
    HostAllocator::free(hrg); HostAllocator::free(hru);
    CudaAllocator::free(dAg); CudaAllocator::free(dAu);
    CudaAllocator::free(dx); CudaAllocator::free(dyg); CudaAllocator::free(dyu);
}

// ─── Main ───────────────────────────────────────────────────

int main()
{
    constexpr int nruns = 500;
    constexpr int M    = C::MOE_INTERMEDIATE_SIZE; // 768
    constexpr int N    = C::HIDDEN_SIZE;           // 2048
    constexpr int tile = C::TILE_ROWS;             // 96

    printf("=== GEMV benchmark (fp32, 1 block, %d threads) ===\n", C::THREADS_PER_BLOCK);
    printf("Config: TILE=%d  M=%d  N=%d\n\n", tile, M, N);

    printf("── Single GEMV (router path) ──\n");
    bench_single("gemv_tile ILP=4", M, N, tile,
        [](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_gemv<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);

    printf("\n── Fused gate+up prefetch (FFN1 path) ──\n");
    bench_fused("fused prefetch ILP=4", M, N, tile,
        [](float *Ag, float *Au, float *x, float *yg, float *yu, int N, int rb, int rc)
        { kern_fused_prefetch<128,4><<<1,128>>>(Ag, Au, x, yg, yu, N, rb, rc); }, nruns);

    printf("\n── Tile sweep (gemv_tile, ILP=4) ──\n");
    for (int t : {48, 96, 192, 384, 768}) {
        if (t > M) continue;
        char name[64];
        snprintf(name, sizeof(name), "tile=%d", t);
        bench_single(name, M, N, t,
            [](float *A, float *x, float *y, int N, int rb, int rc)
            { kern_gemv<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);
    }

    return 0;
}
