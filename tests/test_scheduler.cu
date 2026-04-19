// Test the scheduler + worker doorbell loop with dummy tasks.
//
// 1 OS block (bootstrap pushes tasks, scheduler assigns via doorbells)
// 4 worker blocks (poll doorbell, read task, write expert_id to results, mark ready)
//
// Build:  make test_scheduler
// Run:    ./build/test_scheduler

#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/allocator.cu"
#include "../csrc/queue.cu"

constexpr int NUM_TASKS = 8;
constexpr int NUM_WORKERS = 4;

// Minimal scheduler: push tasks then assign them
__device__ void mini_scheduler(
    TaskQueue<128> *tq, Doorbell *doorbells,
    int *status_queue, int num_workers, int total)
{
    int scheduled = 0;
    while (scheduled < total) {
        int task_idx;
        if (!tq->pop(&task_idx))
            continue;

        // Find ready worker
        int w = -1;
        while (w < 0) {
            for (int i = 0; i < num_workers; i++) {
                int old = atomicExch(&status_queue[i], PROC_BUSY);
                if (old == PROC_READY) { w = i; break; }
            }
        }

        doorbells[w].task_idx = task_idx;
        __threadfence();
        atomicExch(&doorbells[w].ready, 1);
        scheduled++;
    }

    // Send EXIT
    for (int w = 0; w < num_workers; w++) {
        while (atomicExch(&status_queue[w], PROC_BUSY) != PROC_READY) {}
        atomicExch(&doorbells[w].ready, 2);
    }
}

__global__ void test_kernel(
    TaskQueue<128> *tq, Doorbell *doorbells,
    int *status_queue, int *results, int *count)
{
    if (blockIdx.x == 0) {
        // OS block
        int warp = threadIdx.x / 32;
        if (warp == 0 && threadIdx.x == 0) {
            // Push tasks
            for (int i = 0; i < NUM_TASKS; i++) {
                Task t;
                t.expert_id = i * 10;
                t.weight    = 1.0f;
                t.type      = FFN1;
                t.row_begin = 0;
                t.row_count = 96;
                t.slot      = i;
                tq->push(t);
            }
        }
        if (warp == 1 && threadIdx.x == 32) {
            mini_scheduler(tq, doorbells, status_queue, NUM_WORKERS, NUM_TASKS);
        }
    }
    else {
        // Worker blocks
        int worker_id = blockIdx.x - 1;

        if (threadIdx.x == 0)
            atomicExch(&status_queue[worker_id], PROC_READY);

        __shared__ Task current;
        while (true) {
            if (threadIdx.x == 0) {
                while (atomicAdd(&doorbells[worker_id].ready, 0) == 0) {}
                int signal = atomicExch(&doorbells[worker_id].ready, 0);
                if (signal == 2) {
                    current.expert_id = -1;
                } else {
                    int idx = doorbells[worker_id].task_idx;
                    current = tq->entry[idx & 127];
                }
            }
            __syncthreads();

            if (current.expert_id == -1) break;

            // "Execute": write result
            if (threadIdx.x == 0) {
                int slot = atomicAdd(count, 1);
                results[slot] = current.expert_id;
                atomicExch(&status_queue[worker_id], PROC_READY);
            }
            __syncthreads();
        }
    }
}

int main() {
    TaskQueue<128> *tq;
    Doorbell *doorbells;
    int *status_queue, *results, *count;

    CudaAllocator::allocate(&tq, 1);
    CudaAllocator::allocate(&doorbells, NUM_WORKERS);
    CudaAllocator::allocate(&status_queue, NUM_WORKERS);
    CudaAllocator::allocate(&results, NUM_TASKS);
    CudaAllocator::allocate(&count, 1);

    cudaMemset(tq, 0, sizeof(TaskQueue<128>));
    cudaMemset(doorbells, 0, NUM_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, NUM_WORKERS * sizeof(int));
    cudaMemset(results, -1, NUM_TASKS * sizeof(int));
    cudaMemset(count, 0, sizeof(int));

    // 1 OS block + 4 worker blocks = 5 blocks
    test_kernel<<<1 + NUM_WORKERS, 128>>>(tq, doorbells, status_queue, results, count);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int h_count = 0;
    CudaAllocator::copy_to_host(count, &h_count, 1);

    int h_results[NUM_TASKS];
    CudaAllocator::copy_to_host(results, h_results, NUM_TASKS);

    // Check all tasks were processed
    bool ok = (h_count == NUM_TASKS);
    if (!ok) printf("  FAIL: count=%d expected %d\n", h_count, NUM_TASKS);

    // Check all expert_ids present (order may vary)
    bool seen[NUM_TASKS] = {};
    for (int i = 0; i < h_count; i++) {
        int idx = h_results[i] / 10;
        if (idx >= 0 && idx < NUM_TASKS) seen[idx] = true;
    }
    for (int i = 0; i < NUM_TASKS; i++) {
        if (!seen[i]) { ok = false; printf("  FAIL: task %d (expert_id=%d) missing\n", i, i * 10); }
    }

    printf("[Scheduler+Workers] %s  (%d/%d tasks completed)\n",
           ok ? "PASS" : "FAIL", h_count, NUM_TASKS);

    CudaAllocator::free(tq);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);
    CudaAllocator::free(results);
    CudaAllocator::free(count);

    return ok ? 0 : 1;
}
