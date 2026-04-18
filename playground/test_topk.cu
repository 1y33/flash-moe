// Standalone test for flashmoe::topk_kernel<E=128, K=8>.
// Verifies the K ids returned are the K largest of the input.

#include <cstdio>
#include <algorithm>
#include <vector>
#include "../csrc/allocator.cu"
#include "../csrc/tasks/topk.cuh"
#include "bench.cuh"

using flashmoe::topk_kernel;

constexpr int E = 128;
constexpr int K = 8;

int main()
{
    float *h_logits, *h_top_logits;
    int   *h_ids;
    HostAllocator::allocate(&h_logits,     E);
    HostAllocator::allocate(&h_top_logits, K);
    HostAllocator::allocate(&h_ids,        K);

    srand(321);
    HostAllocator::fill_random(h_logits, E);

    float *d_logits, *d_top_logits;
    int   *d_ids;
    CudaAllocator::allocate(&d_logits,     E);
    CudaAllocator::allocate(&d_top_logits, K);
    CudaAllocator::allocate(&d_ids,        K);
    CudaAllocator::copy_to_device(h_logits, d_logits, E);

    topk_kernel<E, K><<<1, 32>>>(d_logits, d_ids, d_top_logits);
    cudaDeviceSynchronize();

    int nruns = 1000;
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; ++r)
        topk_kernel<E, K><<<1, 32>>>(d_logits, d_ids, d_top_logits);
    t.end();

    CudaAllocator::copy_to_host(d_ids,        h_ids,        K);
    CudaAllocator::copy_to_host(d_top_logits, h_top_logits, K);

    // Reference top-K
    std::vector<std::pair<float, int>> pairs(E);
    for (int i = 0; i < E; ++i) pairs[i] = {h_logits[i], i};
    std::sort(pairs.begin(), pairs.end(),
              [](auto &a, auto &b){ return a.first > b.first; });

    std::vector<int> got(h_ids, h_ids + K), want;
    for (int k = 0; k < K; ++k) want.push_back(pairs[k].second);
    std::sort(got.begin(),  got.end());
    std::sort(want.begin(), want.end());
    bool ids_ok = (got == want);

    float max_err = 0.0f;
    for (int k = 0; k < K; ++k)
        max_err = fmaxf(max_err, fabsf(h_top_logits[k] - h_logits[h_ids[k]]));

    float avg_ms = t.elapsed_ms() / nruns;
    printf("\n[topk] E=%d K=%d\n", E, K);
    printf("  ids match (set): %s\n", ids_ok ? "PASS" : "FAIL");
    printf("  top_logits match original: max_err=%.3e  %s\n",
           max_err, (max_err == 0.0f) ? "PASS" : "FAIL");
    printf("  time: %.4f us / launch\n", avg_ms * 1e3f);
    printf("  ids: ");        for (int k = 0; k < K; ++k) printf("%4d ", h_ids[k]);        printf("\n");
    printf("  vals:");        for (int k = 0; k < K; ++k) printf("% .3f ", h_top_logits[k]); printf("\n");

    HostAllocator::free(h_logits); HostAllocator::free(h_top_logits); HostAllocator::free(h_ids);
    CudaAllocator::free(d_logits); CudaAllocator::free(d_top_logits); CudaAllocator::free(d_ids);
    return 0;
}
