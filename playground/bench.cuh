#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        cudaEventRecord(start);
    }

    void end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float elapsed_ms() {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct BenchResult {
    float avg_ms;
    double tflops;
    double bw_gbps;

    void print() {
        printf("  time=%.4f ms  TFLOPS=%.4f  BW=%.2f GB/s\n", avg_ms, tflops, bw_gbps);
    }
};

struct Bench {
    static BenchResult run(double flops, double bytes, int nruns, GpuTimer &timer) {
        float avg_ms = timer.elapsed_ms() / nruns;
        double tflops = (flops / (avg_ms * 1e-3)) / 1e12;
        double bw_gbps = (bytes / (avg_ms * 1e-3)) / 1e9;
        return {avg_ms, tflops, bw_gbps};
    }

    static BenchResult gemv(int M, int N, int nruns, GpuTimer &timer) {
        double flops = 2.0 * M * N;
        double bytes = (double)(M * N + N + M) * sizeof(float);
        return run(flops, bytes, nruns, timer);
    }

    static bool verify(const float *gpu, const float *ref, int count, float tol = 1e-3f) {
        float max_err = 0.0f;
        for (int i = 0; i < count; i++) {
            float err = fabsf(gpu[i] - ref[i]);
            if (err > max_err) max_err = err;
        }
        bool pass = max_err < tol;
        printf("  max_error=%.6f  %s\n", max_err, pass ? "PASS" : "FAIL");
        return pass;
    }
};
