// Standalone test for flashmoe::silu_mul_kernel.
//
// Verifies silu(gate) * up for I=768 (MOE_INTERMEDIATE_SIZE) against a CPU ref.

#include <cstdio>
#include <cmath>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/silu_mul.cuh"
#include "bench.cuh"

using flashmoe::silu_mul_kernel;

static void silu_mul_cpu(const float *g, const float *u, float *o, int I)
{
    for (int i = 0; i < I; ++i)
    {
        float s = g[i] / (1.0f + expf(-g[i]));
        o[i] = s * u[i];
    }
}

int main()
{
    constexpr int I = 768;
    constexpr int TPB = 128;

    float *hg, *hu, *ho, *href;
    HostAllocator::allocate(&hg,   I);
    HostAllocator::allocate(&hu,   I);
    HostAllocator::allocate(&ho,   I);
    HostAllocator::allocate(&href, I);

    srand(7);
    HostAllocator::fill_random(hg, I);
    HostAllocator::fill_random(hu, I);

    silu_mul_cpu(hg, hu, href, I);

    float *dg, *du, *dout;
    CudaAllocator::allocate(&dg,   I);
    CudaAllocator::allocate(&du,   I);
    CudaAllocator::allocate(&dout, I);
    CudaAllocator::copy_to_device(hg, dg, I);
    CudaAllocator::copy_to_device(hu, du, I);

    // warmup
    silu_mul_kernel<float, TPB><<<1, TPB>>>(dg, du, dout, I);
    cudaDeviceSynchronize();

    int nruns = 10000;
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; ++r)
        silu_mul_kernel<float, TPB><<<1, TPB>>>(dg, du, dout, I);
    t.end();

    CudaAllocator::copy_to_host(dout, ho, I);

    double bytes = 3.0 * I * sizeof(float);
    double flops = 4.0 * I;
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    printf("\n[silu_mul] I=%d TPB=%d\n", I, TPB);
    Bench::verify(ho, href, I, 1e-3f);
    res.print();

    HostAllocator::free(hg); HostAllocator::free(hu);
    HostAllocator::free(ho); HostAllocator::free(href);
    CudaAllocator::free(dg); CudaAllocator::free(du); CudaAllocator::free(dout);
    return 0;
}
