// Full kernel test in pure C++ — no Python, no torch, fast compile.
// Tests the real flash_moe_kernel with random weights.
//
// Build:  make test_kernel       (fp16)
//         make test_kernel_fp32  (fp32)
// Run:    ./build/test_kernel

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../csrc/kernel.cu"

#ifndef MODEL_TYPE_FP32
using ModelT = __half;
#else
using ModelT = float;
#endif

template <typename T>
static void fill_random_typed(T *dst, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = (T)((float)rand() / (float)RAND_MAX - 0.5f);
}

template <>
void fill_random_typed<__half>(__half *dst, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = __float2half((float)rand() / (float)RAND_MAX - 0.5f);
}

template <typename T>
static void init_random_device(T *d_ptr, size_t count) {
    T *h_ptr = (T*)malloc(count * sizeof(T));
    fill_random_typed(h_ptr, count);
    cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
    free(h_ptr);
}

int main() {
    namespace C = constants;

    const char *type_name = sizeof(ModelT) == 2 ? "fp16" : "fp32";
    printf("Config: H=%d I=%d E=%d K=%d  dtype=%s\n", C::HIDDEN_SIZE, C::MOE_INTERMEDIATE_SIZE,
           C::NUM_EXPERTS, C::TOP_K, type_name);
    printf("Tiles: FFN1=%d FFN2=%d  Total tasks=%d  Capacity=%d\n",
           C::FFN1_TILES_PER_EXPERT, C::FFN2_TILES_PER_EXPERT, C::TOTAL_TASKS, C::CAPACITY);
    printf("Blocks: %d (1 OS + %d workers)\n\n", C::BLOCKSIZE, C::NUM_WORKERS);

    // Allocate model weights
    FlashMoe<ModelT> model;
    allocate_flashmoe<ModelT, CudaAllocator>(model);

    srand(42);
    for (int e = 0; e < C::NUM_EXPERTS; e++) {
        init_random_device(model.experts[e].gate_proj, C::GATE_PROJ_SIZE);
        init_random_device(model.experts[e].up_proj,   C::UP_PROJ_SIZE);
        init_random_device(model.experts[e].down_proj, C::DOWN_PROJ_SIZE);
    }
    init_random_device(model.router, C::ROUTER_SIZE);

    // Build MoeState (input is ModelT, intermediates/output are float)
    MoeState<ModelT> state;
    CudaAllocator::allocate(&state.input,    C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.output,   C::HIDDEN_SIZE);
    CudaAllocator::allocate(&state.ffn1_out, C::TOP_K * C::MOE_INTERMEDIATE_SIZE);
    CudaAllocator::allocate(&state.ffn1_done, C::TOP_K);

    init_random_device(state.input, C::HIDDEN_SIZE);

    cudaMemset(state.output, 0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(float));

    int ffn1_init[C::TOP_K];
    for (int i = 0; i < C::TOP_K; i++)
        ffn1_init[i] = C::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);

    // Scheduling structures
    TaskQueue<C::CAPACITY> *task_queue;
    Doorbell *doorbells;
    int *status_queue;

    CudaAllocator::allocate(&task_queue, 1);
    CudaAllocator::allocate(&doorbells, C::NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, C::NUM_WORKERS);

    cudaMemset(task_queue, 0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells, 0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));

    // Trace buffer
    DeviceTracer tracer;
    tracer.max_events = 32768;
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    // Warmup
    flash_moe_kernel<ModelT><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue, tracer);
    cudaDeviceSynchronize();

    // Reset state for timed run
    cudaMemset(state.output, 0, C::HIDDEN_SIZE * sizeof(float));
    cudaMemset(state.ffn1_out, 0, C::TOP_K * C::MOE_INTERMEDIATE_SIZE * sizeof(float));
    CudaAllocator::copy_to_device(ffn1_init, state.ffn1_done, C::TOP_K);
    cudaMemset(task_queue, 0, sizeof(TaskQueue<C::CAPACITY>));
    cudaMemset(doorbells, 0, C::NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, C::NUM_WORKERS * sizeof(int));
    TraceBuffer::free(tracer.buf, tracer.count);
    TraceBuffer::allocate(&tracer.buf, &tracer.count, tracer.max_events);

    // Timed run
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    printf("Launching kernel...\n");
    cudaEventRecord(t0);
    flash_moe_kernel<ModelT><<<C::BLOCKSIZE, C::THREADS_PER_BLOCK>>>(
        model, state, task_queue, doorbells, status_queue, tracer);
    cudaEventRecord(t1);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read output
    float h_output[C::HIDDEN_SIZE];
    CudaAllocator::copy_to_host(state.output, h_output, C::HIDDEN_SIZE);

    float norm = 0.0f;
    for (int i = 0; i < C::HIDDEN_SIZE; i++)
        norm += h_output[i] * h_output[i];
    norm = sqrtf(norm);

    printf("Output norm: %.6f\n", norm);
    printf("Output[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    bool nonzero = norm > 1e-6f;
    printf("\n%s (output is %s)\n", nonzero ? "PASS" : "FAIL",
           nonzero ? "nonzero" : "all zeros — something went wrong");
    printf("Kernel time: %.4f ms\n", kernel_ms);

    // Print trace + write JSON for viewer
    TraceBuffer::print(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS);
    TraceBuffer::write_json(tracer.buf, tracer.count, trace_label_names(), TR_NUM_LABELS, "trace.json", TR_FIRST_LEAF);
    TraceBuffer::free(tracer.buf, tracer.count);

    // Cleanup
    free_flashmoe<ModelT, CudaAllocator>(model);
    CudaAllocator::free(state.input);
    CudaAllocator::free(state.output);
    CudaAllocator::free(state.ffn1_out);
    CudaAllocator::free(state.ffn1_done);
    CudaAllocator::free(task_queue);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);

    return nonzero ? 0 : 1;
}
