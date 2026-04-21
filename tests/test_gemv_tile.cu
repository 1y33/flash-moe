// Benchmark gemv_tile variants at persistent kernel tile sizes.
// Single block, TILE_ROWS rows — simulates one worker task.
//
// Build:  make test_gemv_tile
// Run:    ./build/test_gemv_tile

#include <cstdio>
#include <cmath>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/tasks/gemv.cuh"
#include "bench.cuh"

namespace C = constants;

// ─── Shared memory x: all threads cooperatively load x into smem ───

template <typename T, int TPB, int ILP>
__device__ __forceinline__ void gemv_tile_smem(
    const T *__restrict__ A,
    const T *__restrict__ x,
    float   *__restrict__ y,
    float   *__restrict__ x_smem,  // shared memory buffer [N]
    int N, int row_begin, int row_count)
{
    using D = DType<T>;
    constexpr int VEC = D::VEC;
    constexpr int WARPS = TPB / 32;

    // Cooperatively load x into shared memory
    for (int i = threadIdx.x; i < N; i += TPB)
        x_smem[i] = x[i];
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int N_VEC = N / VEC;

    for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS * ILP)
    {
        float acc[ILP];
        #pragma unroll
        for (int i = 0; i < ILP; i++) acc[i] = 0.0f;

        #pragma unroll 4
        for (int j = lane; j < N_VEC; j += 32)
        {
            float b[VEC];
            D::load_vec(x_smem + j * VEC, b);

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    float a[VEC];
                    D::load_vec(A + (size_t)(row_begin + r_base + i) * N + j * VEC, a);
                    acc[i] += D::dot(a, b);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < ILP; i++)
        {
            if (r_base + i < row_count) {
                acc[i] = warp::reduce_sum(acc[i]);
                if (lane == 0)
                    y[row_begin + r_base + i] = acc[i];
            }
        }
    }
}

// ─── Fused gate+up with shared memory x ───

template <typename T, int TPB, int ILP>
__device__ __forceinline__ void gemv_tile_fused_gate_up_smem(
    const T *__restrict__ A_gate,
    const T *__restrict__ A_up,
    const T *__restrict__ x,
    float   *__restrict__ y_gate,
    float   *__restrict__ y_up,
    float   *__restrict__ x_smem,
    int N, int row_begin, int row_count)
{
    using D = DType<T>;
    constexpr int VEC = D::VEC;
    constexpr int WARPS = TPB / 32;

    for (int i = threadIdx.x; i < N; i += TPB)
        x_smem[i] = x[i];
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int N_VEC = N / VEC;

    for (int r_base = wid * ILP; r_base < row_count; r_base += WARPS * ILP)
    {
        float acc_g[ILP], acc_u[ILP];
        #pragma unroll
        for (int i = 0; i < ILP; i++) { acc_g[i] = 0.0f; acc_u[i] = 0.0f; }

        #pragma unroll 4
        for (int j = lane; j < N_VEC; j += 32)
        {
            float b[VEC];
            D::load_vec(x_smem + j * VEC, b);

            #pragma unroll
            for (int i = 0; i < ILP; i++)
            {
                if (r_base + i < row_count) {
                    size_t off = (size_t)(row_begin + r_base + i) * N + j * VEC;
                    float ag[VEC], au[VEC];
                    D::load_vec(A_gate + off, ag);
                    D::load_vec(A_up   + off, au);
                    acc_g[i] += D::dot(ag, b);
                    acc_u[i] += D::dot(au, b);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < ILP; i++)
        {
            if (r_base + i < row_count) {
                acc_g[i] = warp::reduce_sum(acc_g[i]);
                acc_u[i] = warp::reduce_sum(acc_u[i]);
                if (lane == 0) {
                    y_gate[row_begin + r_base + i] = acc_g[i];
                    y_up[row_begin + r_base + i]   = acc_u[i];
                }
            }
        }
    }
}

// ─── Kernel wrappers ────────────────────────────────────────

template <int TPB, int ILP>
__global__ void kern_smem(
    const float *A, const float *x, float *y,
    int N, int row_begin, int row_count)
{
    extern __shared__ float smem[];
    gemv_tile_smem<float, TPB, ILP>(A, x, y, smem, N, row_begin, row_count);
}

template <int TPB, int ILP>
__global__ void kern_ilp(
    const float *A, const float *x, float *y,
    int N, int row_begin, int row_count)
{
    // Use gemv_tile from gemv.cuh (already has ILP)
    flashmoe::gemv_tile<float, TPB, ILP>(A, x, y, N, row_begin, row_count);
}

template <int TPB>
__global__ void kern_gate_up_baseline(
    const float *A_gate, const float *A_up, const float *x,
    float *y_gate, float *y_up,
    int N, int row_begin, int row_count)
{
    flashmoe::gemv_tile<float, TPB>(A_gate, x, y_gate, N, row_begin, row_count);
    flashmoe::gemv_tile<float, TPB>(A_up,   x, y_up,   N, row_begin, row_count);
}

template <int TPB, int ILP>
__global__ void kern_gate_up_fused(
    const float *A_gate, const float *A_up, const float *x,
    float *y_gate, float *y_up,
    int N, int row_begin, int row_count)
{
    flashmoe::gemv_tile_fused_gate_up<float, TPB, ILP>(A_gate, A_up, x, y_gate, y_up, N, row_begin, row_count);
}

template <int TPB, int ILP>
__global__ void kern_gate_up_fused_smem(
    const float *A_gate, const float *A_up, const float *x,
    float *y_gate, float *y_up,
    int N, int row_begin, int row_count)
{
    extern __shared__ float smem[];
    gemv_tile_fused_gate_up_smem<float, TPB, ILP>(A_gate, A_up, x, y_gate, y_up, smem, N, row_begin, row_count);
}

template <int TPB, int ILP>
__global__ void kern_gate_up_fused_prefetch(
    const float *A_gate, const float *A_up, const float *x,
    float *y_gate, float *y_up,
    int N, int row_begin, int row_count)
{
    flashmoe::gemv_tile_fused_gate_up_prefetch<float, TPB, ILP>(A_gate, A_up, x, y_gate, y_up, N, row_begin, row_count);
}

// ─── CPU reference ──────────────────────────────────────────

static void gemv_cpu(const float *A, const float *x, float *y,
                     int N, int row_begin, int row_count)
{
    for (int r = 0; r < row_count; r++) {
        int row = row_begin + r;
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += A[row * N + j] * x[j];
        y[row] = acc;
    }
}

// ─── Single GEMV benchmark ──────────────────────────────────

template <typename KernelFn>
static void bench_single(const char *name, int M, int N, int tile,
                          KernelFn kernel_fn, int nruns)
{
    float *hA, *hx, *hy, *hy_ref;
    HostAllocator::allocate(&hA, M * N);
    HostAllocator::allocate(&hx, N);
    HostAllocator::allocate(&hy, M);
    HostAllocator::allocate(&hy_ref, M);

    srand(42);
    HostAllocator::fill_random(hA, M * N);
    HostAllocator::fill_random(hx, N);
    memset(hy_ref, 0, M * sizeof(float));
    gemv_cpu(hA, hx, hy_ref, N, 0, tile);

    float *dA, *dx, *dy;
    CudaAllocator::allocate(&dA, M * N);
    CudaAllocator::allocate(&dx, N);
    CudaAllocator::allocate(&dy, M);
    CudaAllocator::copy_to_device(hA, dA, M * N);
    CudaAllocator::copy_to_device(hx, dx, N);

    cudaMemset(dy, 0, M * sizeof(float));
    kernel_fn(dA, dx, dy, N, 0, tile);
    cudaDeviceSynchronize();

    cudaMemset(dy, 0, M * sizeof(float));
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; r++)
        kernel_fn(dA, dx, dy, N, 0, tile);
    t.end();

    CudaAllocator::copy_to_host(dy, hy, M);

    double flops = 2.0 * tile * N;
    double bytes = (double)(tile * N + N + tile) * sizeof(float);
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    float max_err = 0;
    for (int i = 0; i < tile; i++) {
        float e = fabsf(hy[i] - hy_ref[i]);
        if (e > max_err) max_err = e;
    }
    bool pass = max_err < 1e-3f;
    printf("  %-32s err=%.1e %s  %.4fms  %.4f TFLOPS  %6.1f GB/s\n",
           name, max_err, pass ? "OK" : "FAIL", res.avg_ms, res.tflops, res.bw_gbps);

    HostAllocator::free(hA); HostAllocator::free(hx);
    HostAllocator::free(hy); HostAllocator::free(hy_ref);
    CudaAllocator::free(dA); CudaAllocator::free(dx); CudaAllocator::free(dy);
}

// ─── Fused gate+up benchmark ────────────────────────────────

template <typename KernelFn>
static void bench_fused(const char *name, int M, int N, int tile,
                         KernelFn kernel_fn, int nruns)
{
    float *hA_g, *hA_u, *hx, *hy_g, *hy_u, *href_g, *href_u;
    HostAllocator::allocate(&hA_g, M * N);
    HostAllocator::allocate(&hA_u, M * N);
    HostAllocator::allocate(&hx, N);
    HostAllocator::allocate(&hy_g, M);
    HostAllocator::allocate(&hy_u, M);
    HostAllocator::allocate(&href_g, M);
    HostAllocator::allocate(&href_u, M);

    srand(42);
    HostAllocator::fill_random(hA_g, M * N);
    HostAllocator::fill_random(hA_u, M * N);
    HostAllocator::fill_random(hx, N);
    memset(href_g, 0, M * sizeof(float));
    memset(href_u, 0, M * sizeof(float));
    gemv_cpu(hA_g, hx, href_g, N, 0, tile);
    gemv_cpu(hA_u, hx, href_u, N, 0, tile);

    float *dA_g, *dA_u, *dx, *dy_g, *dy_u;
    CudaAllocator::allocate(&dA_g, M * N);
    CudaAllocator::allocate(&dA_u, M * N);
    CudaAllocator::allocate(&dx, N);
    CudaAllocator::allocate(&dy_g, M);
    CudaAllocator::allocate(&dy_u, M);
    CudaAllocator::copy_to_device(hA_g, dA_g, M * N);
    CudaAllocator::copy_to_device(hA_u, dA_u, M * N);
    CudaAllocator::copy_to_device(hx, dx, N);

    cudaMemset(dy_g, 0, M * sizeof(float));
    cudaMemset(dy_u, 0, M * sizeof(float));
    kernel_fn(dA_g, dA_u, dx, dy_g, dy_u, N, 0, tile);
    cudaDeviceSynchronize();

    cudaMemset(dy_g, 0, M * sizeof(float));
    cudaMemset(dy_u, 0, M * sizeof(float));
    GpuTimer t; t.begin();
    for (int r = 0; r < nruns; r++)
        kernel_fn(dA_g, dA_u, dx, dy_g, dy_u, N, 0, tile);
    t.end();

    CudaAllocator::copy_to_host(dy_g, hy_g, M);
    CudaAllocator::copy_to_host(dy_u, hy_u, M);

    double flops = 2.0 * 2 * tile * N;
    double bytes = (double)(2 * tile * N + N + 2 * tile) * sizeof(float);
    BenchResult res = Bench::run(flops, bytes, nruns, t);

    float max_err = 0;
    for (int i = 0; i < tile; i++) {
        float e1 = fabsf(hy_g[i] - href_g[i]);
        float e2 = fabsf(hy_u[i] - href_u[i]);
        if (e1 > max_err) max_err = e1;
        if (e2 > max_err) max_err = e2;
    }
    bool pass = max_err < 1e-3f;
    printf("  %-32s err=%.1e %s  %.4fms  %.4f TFLOPS  %6.1f GB/s\n",
           name, max_err, pass ? "OK" : "FAIL", res.avg_ms, res.tflops, res.bw_gbps);

    HostAllocator::free(hA_g); HostAllocator::free(hA_u);
    HostAllocator::free(hx); HostAllocator::free(hy_g); HostAllocator::free(hy_u);
    HostAllocator::free(href_g); HostAllocator::free(href_u);
    CudaAllocator::free(dA_g); CudaAllocator::free(dA_u);
    CudaAllocator::free(dx); CudaAllocator::free(dy_g); CudaAllocator::free(dy_u);
}

int main()
{
    int nruns = 500;
    int M = C::MOE_INTERMEDIATE_SIZE; // 768
    int N = C::HIDDEN_SIZE;           // 2048
    int tile = C::TILE_ROWS;          // 96
    int smem_bytes = N * sizeof(float); // 8KB for x

    printf("=== gemv_tile benchmark (1 block, 128 threads) ===\n");
    printf("Config: TILE=%d  M=%d  N=%d  smem for x=%d bytes\n\n", tile, M, N, smem_bytes);

    // ── Single GEMV (96 x 2048) ──
    printf("── Single GEMV %dx%d ──\n", tile, N);

    bench_single("ILP=4 (current)", M, N, tile,
        [](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_ilp<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("ILP=4 + smem x", M, N, tile,
        [=](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_smem<128,4><<<1,128,smem_bytes>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("ILP=8 + smem x", M, N, tile,
        [=](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_smem<128,8><<<1,128,smem_bytes>>>(A, x, y, N, rb, rc); }, nruns);

    // ── Gate+Up fused (96 x 2048) ──
    printf("\n── Gate+Up fused %dx%d ──\n", tile, N);

    bench_fused("fused ILP=4 (current)", M, N, tile,
        [](float *Ag, float *Au, float *x, float *yg, float *yu, int N, int rb, int rc)
        { kern_gate_up_fused<128,4><<<1,128>>>(Ag, Au, x, yg, yu, N, rb, rc); }, nruns);

    bench_fused("fused ILP=4 + smem x", M, N, tile,
        [=](float *Ag, float *Au, float *x, float *yg, float *yu, int N, int rb, int rc)
        { kern_gate_up_fused_smem<128,4><<<1,128,smem_bytes>>>(Ag, Au, x, yg, yu, N, rb, rc); }, nruns);

    bench_fused("fused ILP=4 prefetch", M, N, tile,
        [](float *Ag, float *Au, float *x, float *yg, float *yu, int N, int rb, int rc)
        { kern_gate_up_fused_prefetch<128,4><<<1,128>>>(Ag, Au, x, yg, yu, N, rb, rc); }, nruns);

    bench_fused("fused ILP=2 + smem x", M, N, tile,
        [=](float *Ag, float *Au, float *x, float *yg, float *yu, int N, int rb, int rc)
        { kern_gate_up_fused_smem<128,2><<<1,128,smem_bytes>>>(Ag, Au, x, yg, yu, N, rb, rc); }, nruns);

    // ── Bigger tiles with smem ──
    printf("\n── Bigger tiles + smem x ──\n");

    bench_single("tile=192 ILP=4 + smem", M, N, 192,
        [=](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_smem<128,4><<<1,128,smem_bytes>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("tile=384 ILP=4 + smem", M, N, 384,
        [=](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_smem<128,4><<<1,128,smem_bytes>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("tile=768 ILP=4 + smem", M, N, 768,
        [=](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_smem<128,4><<<1,128,smem_bytes>>>(A, x, y, N, rb, rc); }, nruns);

    // Compare: no smem at same tile sizes
    printf("\n── Bigger tiles NO smem (for comparison) ──\n");

    bench_single("tile=192 ILP=4", M, N, 192,
        [](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_ilp<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("tile=384 ILP=4", M, N, 384,
        [](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_ilp<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);

    bench_single("tile=768 ILP=4", M, N, 768,
        [](float *A, float *x, float *y, int N, int rb, int rc)
        { kern_ilp<128,4><<<1,128>>>(A, x, y, N, rb, rc); }, nruns);

    return 0;
}
