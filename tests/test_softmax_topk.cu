// Standalone test for flashmoe::softmax_topk_kernel<E=128, K=8>.
//
// Verifies that:
//   1. The K returned ids are indeed the K largest logits.
//   2. weights[k] == softmax(logits[ids[k]] - max) and sum == 1.
//
// Also times the kernel.

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/softmax_topk.cuh"
#include "bench.cuh"

using flashmoe::softmax_topk_kernel;

constexpr int E = 128;
constexpr int K = 8;

int main()
{
    float *h_logits, *h_weights;
    int   *h_ids;
    HostAllocator::allocate(&h_logits,  E);
    HostAllocator::allocate(&h_weights, K);
    HostAllocator::allocate(&h_ids,     K);

    srand(123);
    HostAllocator::fill_random(h_logits, E);

    float *d_logits, *d_weights;
    int   *d_ids;
    CudaAllocator::allocate(&d_logits,  E);
    CudaAllocator::allocate(&d_weights, K);
    CudaAllocator::allocate(&d_ids,     K);
    CudaAllocator::copy_to_device(h_logits, d_logits, E);

    // warmup
    softmax_topk_kernel<E, K><<<1, 32>>>(d_logits, d_ids, d_weights);
    cudaDeviceSynchronize();

    int nruns = 1000;
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; ++r)
        softmax_topk_kernel<E, K><<<1, 32>>>(d_logits, d_ids, d_weights);
    t.end();

    CudaAllocator::copy_to_host(d_ids,     h_ids,     K);
    CudaAllocator::copy_to_host(d_weights, h_weights, K);

    // ---- reference: sort logits descending, take first K ----
    std::vector<std::pair<float, int>> pairs(E);
    for (int i = 0; i < E; ++i) pairs[i] = {h_logits[i], i};
    std::sort(pairs.begin(), pairs.end(),
              [](auto &a, auto &b){ return a.first > b.first; });

    // check ids as a SET (order of ties may differ — our impl tie-breaks on
    // smaller lane, which gives a deterministic but not sorted order).
    std::vector<int> got_ids(h_ids, h_ids + K);
    std::vector<int> want_ids;
    for (int k = 0; k < K; ++k) want_ids.push_back(pairs[k].second);
    std::sort(got_ids.begin(),  got_ids.end());
    std::sort(want_ids.begin(), want_ids.end());

    bool ids_ok = (got_ids == want_ids);

    // softmax reference on the K winners
    float m = -1e30f;
    for (int k = 0; k < K; ++k) m = fmaxf(m, h_logits[h_ids[k]]);
    float s = 0.0f;
    float ref_w[K];
    for (int k = 0; k < K; ++k) { ref_w[k] = expf(h_logits[h_ids[k]] - m); s += ref_w[k]; }
    for (int k = 0; k < K; ++k) ref_w[k] /= s;

    float max_err = 0.0f;
    for (int k = 0; k < K; ++k)
        max_err = fmaxf(max_err, fabsf(ref_w[k] - h_weights[k]));

    float wsum = 0.0f;
    for (int k = 0; k < K; ++k) wsum += h_weights[k];

    float avg_ms = t.elapsed_ms() / nruns;
    printf("\n[softmax_topk] E=%d K=%d\n", E, K);
    printf("  ids match (set): %s\n", ids_ok ? "PASS" : "FAIL");
    printf("  weight max_err : %.3e  %s\n", max_err, (max_err < 1e-5f) ? "PASS" : "FAIL");
    printf("  sum(weights)   : %.6f\n", wsum);
    printf("  time           : %.4f us / launch\n", avg_ms * 1e3f);

    printf("  ids:    ");
    for (int k = 0; k < K; ++k) printf("%4d ", h_ids[k]); printf("\n");
    printf("  weights:");
    for (int k = 0; k < K; ++k) printf("%.3f ", h_weights[k]); printf("\n");

    HostAllocator::free(h_logits); HostAllocator::free(h_weights); HostAllocator::free(h_ids);
    CudaAllocator::free(d_logits); CudaAllocator::free(d_weights); CudaAllocator::free(d_ids);
    return 0;
}
