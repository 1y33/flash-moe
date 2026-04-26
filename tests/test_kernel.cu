// Full kernel test — no Python, no torch, fast compile.
//
// Build:  make test_kernel       (fp16 weights + fp16 activations)
//         make test_kernel_fp32  (fp32 weights + fp32 activations)
// Run:    ./build/test_kernel

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../csrc/kernel.cu"

#ifndef MODEL_TYPE_FP32
using T = __half;
#else
using T = float;
#endif

template <typename U>
static void init_random_device(U *d_ptr, size_t count) {
    float *tmp = (float *)malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++)
        tmp[i] = (float)rand() / RAND_MAX - 0.5f;

    U *h = (U *)malloc(count * sizeof(U));
    for (size_t i = 0; i < count; i++)
        h[i] = (U)tmp[i];

    cudaMemcpy(d_ptr, h, count * sizeof(U), cudaMemcpyHostToDevice);
    free(tmp);
    free(h);
}

int main() {
    namespace C = constants;

    const char *dtype = sizeof(T) == 2 ? "fp16" : "fp32";
    printf("Config: H=%d I=%d E=%d K=%d  dtype=%s\n",
           C::HIDDEN_SIZE, C::MOE_INTERMEDIATE_SIZE, C::NUM_EXPERTS, C::TOP_K, dtype);
    printf("Tiles: FFN1=%d FFN2=%d  Total=%d  Capacity=%d\n",
           C::FFN1_TILES_PER_EXPERT, C::FFN2_TILES_PER_EXPERT, C::TOTAL_TASKS, C::CAPACITY);
    printf("Blocks: %d (1 OS + %d workers)\n\n", C::BLOCKSIZE, C::NUM_WORKERS);

    // Model weights
    FlashMoe<T> model;
    allocate_flashmoe<T, CudaAllocator>(model);

    srand(42);
    for (int e = 0; e < C::NUM_EXPERTS; e++) {
        init_random_device(model.experts[e].gate_proj, C::GATE_PROJ_SIZE);
        init_random_device(model.experts[e].up_proj,   C::UP_PROJ_SIZE);
        init_random_device(model.experts[e].down_proj, C::DOWN_PROJ_SIZE);
    }
    init_random_device(model.router, C::ROUTER_SIZE);

    // State: T for weights and activations
    MoeState<T, T> state;
    CudaAllocator::allocate(&state.input,    C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.output,   C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.ffn1_out, C::TOP_K * C::MOE_INTERMEDIATE_SIZE);
    CudaAllocator::allocate(&state.ffn1_done, C::TOP_K);
    CudaAllocator::allocate(&state.logits, C::NUM_EXPERTS);
    CudaAllocator::allocate(&state.router_done, 1);

    init_random_device(state.input, C::HIDDEN_SIZE);
    cudaMemset(state.output,   0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(T));
    cudaMemset(state.logits, 0, C::NUM_EXPERTS * sizeof(float));
    cudaMemset(state.router_done, 0, sizeof(int));

    int ffn1_init[C::TOP_K];
    for (int i = 0; i < C::TOP_K; i++)
        ffn1_init[i] = C::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);

    // Scheduling
    TaskQueue<C::CAPACITY> *task_queue;
    Doorbell *doorbells;
    int *status_queue;

    CudaAllocator::allocate(&task_queue, 1);
    CudaAllocator::allocate(&doorbells, C::NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, C::NUM_WORKERS);

    cudaMemset(task_queue,   0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells,    0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));

    // Trace
    DeviceTracer tracer;
    tracer.max_events = 32768;
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    // Warmup
    flash_moe_kernel<T, T><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue, tracer);
    cudaDeviceSynchronize();

    // Reset for timed run
    cudaMemset(state.output,   0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(T));
    cudaMemset(state.logits, 0, C::NUM_EXPERTS * sizeof(float));
    cudaMemset(state.router_done, 0, sizeof(int));
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);
    cudaMemset(task_queue,   0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells,    0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));
    TraceBuffer::free(tracer.buf, tracer.count);
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    // Timed run
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    printf("Launching kernel...\n");
    cudaEventRecord(t0);
    flash_moe_kernel<T, T><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue, tracer);
    cudaEventRecord(t1);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read output (always fp32)
    float h_out[C::HIDDEN_SIZE];
    CudaAllocator::copy_to_host(state.output, h_out, C::HIDDEN_SIZE);

    float norm = 0.0f;
    for (int i = 0; i < C::HIDDEN_SIZE; i++)
        norm += h_out[i] * h_out[i];
    norm = sqrtf(norm);

    printf("Output norm: %.6f\n", norm);
    printf("Output[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    bool nonzero = norm > 1e-6f;
    printf("\n%s (output is %s)\n", nonzero ? "PASS" : "FAIL",
           nonzero ? "nonzero" : "all zeros — something went wrong");
    printf("Kernel time: %.4f ms\n", kernel_ms);

    TraceBuffer::print(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS);
    TraceBuffer::write_json(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS,
                            "trace.json", TR_FIRST_LEAF);
    TraceBuffer::free(tracer.buf, tracer.count);

    // Cleanup
    free_flashmoe<T, CudaAllocator>(model);
    CudaAllocator::free(state.input);
    CudaAllocator::free(state.output);
    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(state.logits);
    CudaAllocator::free(state.router_done);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);

    return nonzero ? 0 : 1;
}
