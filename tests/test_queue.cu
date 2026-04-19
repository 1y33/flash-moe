// Test the TaskQueue push/pop and Doorbell mechanism.
//
// Build:  make test_queue
// Run:    ./build/test_queue

#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/utils/allocator.cuh"
#include "../csrc/queue.cu"

// Test 1: single-producer push, single-consumer pop
__global__ void test_push_pop(TaskQueue<128> *q, int *results) {
    if (threadIdx.x == 0) {
        // Push 8 tasks
        for (int i = 0; i < 8; i++) {
            Task t;
            t.expert_id = i;
            t.weight    = (float)i * 0.1f;
            t.type      = FFN1;
            t.row_begin = i * 96;
            t.row_count = 96;
            t.slot      = i;
            q->push(t);
        }

        // Pop them all
        for (int i = 0; i < 8; i++) {
            int idx;
            bool ok = q->pop(&idx);
            if (ok) {
                Task t = q->entry[idx & 127];
                results[i] = t.expert_id;
            } else {
                results[i] = -1;
            }
        }

        // Should fail (empty)
        int idx;
        results[8] = q->pop(&idx) ? 1 : 0;
    }
}

// Test 2: doorbell signal/poll (both blocks in one kernel so they run concurrently)
__global__ void test_doorbell(Doorbell *db, int *results) {
    if (blockIdx.x == 0) {
        // Writer
        if (threadIdx.x == 0) {
            db[0].task_idx = 42;
            __threadfence();
            atomicExch(&db[0].ready, 1);
        }
    }
    else {
        // Reader
        if (threadIdx.x == 0) {
            while (atomicAdd(&db[0].ready, 0) == 0) {}
            int signal = atomicExch(&db[0].ready, 0);
            results[0] = signal;
            results[1] = db[0].task_idx;
        }
    }
}

int main() {
    int pass = 0, fail = 0;

    // --- Test 1: Push/Pop ---
    {
        TaskQueue<128> *q;
        int *results;
        CudaAllocator::allocate(&q, 1);
        CudaAllocator::allocate(&results, 16);
        cudaMemset(q, 0, sizeof(TaskQueue<128>));
        cudaMemset(results, -1, 16 * sizeof(int));

        test_push_pop<<<1, 1>>>(q, results);
        cudaDeviceSynchronize();

        int h[16];
        CudaAllocator::copy_to_host(results, h, 16);

        bool ok = true;
        for (int i = 0; i < 8; i++) {
            if (h[i] != i) { ok = false; printf("  FAIL: pop[%d] = %d, expected %d\n", i, h[i], i); }
        }
        if (h[8] != 0) { ok = false; printf("  FAIL: pop on empty returned true\n"); }

        printf("[Push/Pop] %s\n", ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;

        CudaAllocator::free(q);
        CudaAllocator::free(results);
    }

    // --- Test 2: Doorbell ---
    {
        Doorbell *db;
        int *results;
        CudaAllocator::allocate(&db, 1);
        CudaAllocator::allocate(&results, 2);
        cudaMemset(db, 0, sizeof(Doorbell));
        cudaMemset(results, 0, 2 * sizeof(int));

        // 2 blocks in one kernel: block 0 = writer, block 1 = reader
        test_doorbell<<<2, 1>>>(db, results);
        cudaDeviceSynchronize();

        int h[2];
        CudaAllocator::copy_to_host(results, h, 2);

        bool ok = (h[0] == 1 && h[1] == 42);
        if (!ok) printf("  FAIL: signal=%d task_idx=%d\n", h[0], h[1]);
        printf("[Doorbell] %s\n", ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;

        CudaAllocator::free(db);
        CudaAllocator::free(results);
    }

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail;
}
