// Standalone test for flashmoe::gemv_kernel over the three MoE GEMV shapes.
//
//   router  : M=128,  N=2048
//   gate/up : M=768,  N=2048
//   down    : M=2048, N=768
//
// Verifies vs CPU reference, reports ms / TFLOPS / GB/s.

#include <cstdio>
#include <cmath>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/gemv.cuh"
#include "bench.cuh"

using flashmoe::gemv_kernel;

static void gemv_cpu(const float *A, const float *x, float *y, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += A[i * N + j] * x[j];
        y[i] = acc;
    }
}

template <int TPB>
static void run_case(const char *name, int M, int N, int nruns)
{
    float *hA, *hx, *hy, *hy_ref;
    HostAllocator::allocate(&hA,     M * N);
    HostAllocator::allocate(&hx,     N);
    HostAllocator::allocate(&hy,     M);
    HostAllocator::allocate(&hy_ref, M);

    srand(42);
    HostAllocator::fill_random(hA, M * N);
    HostAllocator::fill_random(hx, N);

    gemv_cpu(hA, hx, hy_ref, M, N);

    float *dA, *dx, *dy;
    CudaAllocator::allocate(&dA, M * N);
    CudaAllocator::allocate(&dx, N);
    CudaAllocator::allocate(&dy, M);
    CudaAllocator::copy_to_device(hA, dA, M * N);
    CudaAllocator::copy_to_device(hx, dx, N);

    constexpr int WARPS = TPB / 32;
    dim3 block(TPB);
    dim3 grid((M + WARPS - 1) / WARPS);

    // warmup
    gemv_kernel<float, TPB><<<grid, block>>>(dA, dx, dy, M, N);
    cudaDeviceSynchronize();

    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; ++r)
        gemv_kernel<float, TPB><<<grid, block>>>(dA, dx, dy, M, N);
    t.end();

    CudaAllocator::copy_to_host(dy, hy, M);
    BenchResult res = Bench::gemv(M, N, nruns, t);
    printf("\n[%s] M=%d N=%d  TPB=%d\n", name, M, N, TPB);
    Bench::verify(hy, hy_ref, M);
    res.print();

    HostAllocator::free(hA); HostAllocator::free(hx);
    HostAllocator::free(hy); HostAllocator::free(hy_ref);
    CudaAllocator::free(dA); CudaAllocator::free(dx); CudaAllocator::free(dy);
}

int main()
{
    int nruns = 200;
    run_case<128>("router",  128,  2048, nruns);
    run_case<128>("gate_up", 768,  2048, nruns);
    run_case<128>("down",    2048, 768,  nruns);
    run_case<256>("router",  128,  2048, nruns);
    run_case<256>("gate_up", 768,  2048, nruns);
    run_case<256>("down",    2048, 768,  nruns);
    return 0;
}
