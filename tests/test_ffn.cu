// Standalone test for flashmoe::ffn1_kernel and ffn2_kernel.
//
//   FFN1: act[I] = silu(x @ gate_w^T) * (x @ up_w^T)
//   FFN2: y[H]  += scale * (act @ down_w^T)
//
// Shapes (Qwen3 MoE):
//   H = 2048, I = 768.

#include <cstdio>
#include <cmath>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/ffn.cuh"
#include "bench.cuh"

using flashmoe::ffn1_kernel;
using flashmoe::ffn2_kernel;

static void gemv_cpu(const float *A, const float *x, float *y, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += A[i * N + j] * x[j];
        y[i] = acc;
    }
}

int main()
{
    constexpr int H = 2048;
    constexpr int I = 768;
    constexpr int TPB = 128;

    // ---------------- FFN1 ----------------
    float *hx, *hg, *hu, *hact, *href;
    HostAllocator::allocate(&hx,   H);
    HostAllocator::allocate(&hg,   I * H);
    HostAllocator::allocate(&hu,   I * H);
    HostAllocator::allocate(&hact, I);
    HostAllocator::allocate(&href, I);

    srand(11);
    HostAllocator::fill_random(hx, H);
    HostAllocator::fill_random(hg, I * H);
    HostAllocator::fill_random(hu, I * H);

    // CPU reference
    {
        float *g_tmp, *u_tmp;
        HostAllocator::allocate(&g_tmp, I);
        HostAllocator::allocate(&u_tmp, I);
        gemv_cpu(hg, hx, g_tmp, I, H);
        gemv_cpu(hu, hx, u_tmp, I, H);
        for (int i = 0; i < I; ++i)
        {
            float s = g_tmp[i] / (1.0f + expf(-g_tmp[i]));
            href[i] = s * u_tmp[i];
        }
        HostAllocator::free(g_tmp); HostAllocator::free(u_tmp);
    }

    float *dx, *dg, *du, *dgbuf, *dubuf, *dact;
    CudaAllocator::allocate(&dx,   H);
    CudaAllocator::allocate(&dg,   I * H);
    CudaAllocator::allocate(&du,   I * H);
    CudaAllocator::allocate(&dgbuf, I);
    CudaAllocator::allocate(&dubuf, I);
    CudaAllocator::allocate(&dact,  I);
    CudaAllocator::copy_to_device(hx, dx, H);
    CudaAllocator::copy_to_device(hg, dg, I * H);
    CudaAllocator::copy_to_device(hu, du, I * H);

    // warmup
    ffn1_kernel<TPB><<<1, TPB>>>(dx, dg, du, dgbuf, dubuf, dact, H, I);
    cudaDeviceSynchronize();

    int nruns = 500;
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; ++r)
        ffn1_kernel<TPB><<<1, TPB>>>(dx, dg, du, dgbuf, dubuf, dact, H, I);
    t.end();
    CudaAllocator::copy_to_host(dact, hact, I);

    // rough BW/FLOP model
    double bytes = (2.0 * I * H + 2.0 * I + H) * sizeof(float);
    double flops = 2.0 * (2.0 * I * H) + 4.0 * I;
    BenchResult r1 = Bench::run(flops, bytes, nruns, t);
    printf("\n[FFN1] H=%d I=%d TPB=%d\n", H, I, TPB);
    Bench::verify(hact, href, I, 5e-3f);  // silu*mul * accumulated fma rounding
    r1.print();

    // ---------------- FFN2 ----------------
    float *hd, *hy, *hy_ref;
    HostAllocator::allocate(&hd,     H * I);
    HostAllocator::allocate(&hy,     H);
    HostAllocator::allocate(&hy_ref, H);
    HostAllocator::fill_random(hd, H * I);
    for (int i = 0; i < H; ++i) hy_ref[i] = 0.0f;

    const float scale = 0.25f;
    // CPU ref: y = scale * (down @ act)
    for (int i = 0; i < H; ++i)
    {
        float acc = 0.0f;
        for (int j = 0; j < I; ++j) acc += hd[i * I + j] * href[j];
        hy_ref[i] = scale * acc;
    }

    float *dd, *dy;
    CudaAllocator::allocate(&dd, H * I);
    CudaAllocator::allocate(&dy, H);
    CudaAllocator::copy_to_device(hd, dd, H * I);
    cudaMemset(dy, 0, H * sizeof(float));

    // feed the REFERENCE act into FFN2 so we isolate FFN2's correctness
    // from FFN1 output noise.
    float *dact_ref;
    CudaAllocator::allocate(&dact_ref, I);
    CudaAllocator::copy_to_device(href, dact_ref, I);

    ffn2_kernel<TPB><<<1, TPB>>>(dact_ref, dd, dy, H, I, scale);
    cudaDeviceSynchronize();

    // For timing we reset dy each iter — otherwise we'd accumulate nruns times.
    GpuTimer t2; t2.begin();
    for (int r = 0; r < nruns; ++r)
    {
        cudaMemsetAsync(dy, 0, H * sizeof(float));
        ffn2_kernel<TPB><<<1, TPB>>>(dact_ref, dd, dy, H, I, scale);
    }
    t2.end();
    CudaAllocator::copy_to_host(dy, hy, H);

    BenchResult r2 = Bench::gemv(H, I, nruns, t2);
    printf("\n[FFN2] H=%d I=%d TPB=%d scale=%.2f\n", H, I, TPB, scale);
    Bench::verify(hy, hy_ref, H, 5e-3f);
    r2.print();

    HostAllocator::free(hx); HostAllocator::free(hg); HostAllocator::free(hu);
    HostAllocator::free(hact); HostAllocator::free(href);
    HostAllocator::free(hd); HostAllocator::free(hy); HostAllocator::free(hy_ref);
    CudaAllocator::free(dx); CudaAllocator::free(dg); CudaAllocator::free(du);
    CudaAllocator::free(dgbuf); CudaAllocator::free(dubuf); CudaAllocator::free(dact);
    CudaAllocator::free(dd); CudaAllocator::free(dy); CudaAllocator::free(dact_ref);
    return 0;
}
