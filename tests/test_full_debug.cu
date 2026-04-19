// Debug test for the full kernel scheduling loop.
// Uses the real OS + Worker structs but with printf to trace what happens.
// Starts small (4 workers) to isolate hangs.
//
// Build:  make test_full_debug
// Run:    ./build/test_full_debug

#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/allocator.cu"
#include "../csrc/flashmoe.cuh"
#include "../csrc/queue.cu"

// We don't run real FFN — just test the scheduling plumbing.
// Stub executors that just printf and return.

constexpr int DBG_WORKERS = 4;
constexpr int DBG_CAPACITY = 256;  // bigger than 128 to avoid overflow

// --- Instrumented bootstrap: push a few FFN1 tasks ---
__device__ void dbg_bootstrap(TaskQueue<DBG_CAPACITY> *tq) {
    if (threadIdx.x == 0) {
        int count = 0;
        for (int k = 0; k < constants::TOP_K; k++) {
            int rows_left = constants::MOE_INTERMEDIATE_SIZE;
            int row = 0;
            while (rows_left > 0) {
                int chunk = min(constants::TILE_ROWS, rows_left);
                Task t;
                t.expert_id = k;
                t.weight    = 1.0f;
                t.type      = FFN1;
                t.row_begin = row;
                t.row_count = chunk;
                t.slot      = k;
                tq->push(t);
                count++;
                row       += chunk;
                rows_left -= chunk;
            }
        }
        printf("[Bootstrap] pushed %d FFN1 tasks\n", count);
    }
}

// --- Instrumented scheduler ---
__device__ void dbg_scheduler(
    TaskQueue<DBG_CAPACITY> *tq, Doorbell *doorbells,
    int *status_queue, int num_workers, int total_tasks)
{
    int scheduled = 0;
    printf("[Scheduler] starting, expecting %d tasks\n", total_tasks);

    while (scheduled < total_tasks) {
        int task_idx;
        if (!tq->pop(&task_idx))
            continue;

        // Find ready worker
        int w = -1;
        int spins = 0;
        while (w < 0) {
            for (int i = 0; i < num_workers; i++) {
                int old = atomicExch(&status_queue[i], PROC_BUSY);
                if (old == PROC_READY) { w = i; break; }
            }
            spins++;
            if (spins > 1000000) {
                printf("[Scheduler] STUCK finding worker for task %d (scheduled=%d)\n",
                       task_idx, scheduled);
                return;
            }
        }

        doorbells[w].task_idx = task_idx;
        __threadfence();
        atomicExch(&doorbells[w].ready, 1);
        scheduled++;

        if (scheduled % 10 == 0 || scheduled <= 5)
            printf("[Scheduler] scheduled=%d/%d (task_idx=%d -> worker %d)\n",
                   scheduled, total_tasks, task_idx, w);
    }

    printf("[Scheduler] all %d tasks scheduled, sending EXIT\n", scheduled);

    for (int w = 0; w < num_workers; w++) {
        int spins = 0;
        while (atomicExch(&status_queue[w], PROC_BUSY) != PROC_READY) {
            spins++;
            if (spins > 1000000) {
                printf("[Scheduler] STUCK waiting for worker %d to become READY for EXIT\n", w);
                return;
            }
        }
        atomicExch(&doorbells[w].ready, 2);
        printf("[Scheduler] sent EXIT to worker %d\n", w);
    }
    printf("[Scheduler] done\n");
}

// --- Instrumented worker ---
__global__ void dbg_kernel(
    TaskQueue<DBG_CAPACITY> *tq, Doorbell *doorbells,
    int *status_queue, int *ffn1_done, int *task_count)
{
    if (blockIdx.x == 0) {
        // OS block
        int warp = threadIdx.x / 32;
        if (warp == 0) {
            dbg_bootstrap(tq);
        }
        __syncthreads(); // ensure bootstrap done before scheduler starts
        if (warp == 1 && threadIdx.x == 32) {
            // Only FFN1 tasks for now (no fan-in → no FFN2)
            int total = constants::TOP_K * constants::FFN1_TILES_PER_EXPERT;
            dbg_scheduler(tq, doorbells, status_queue, DBG_WORKERS, total);
        }
    }
    else {
        // Worker
        int wid = blockIdx.x - 1;
        __shared__ Task current;
        __shared__ int done;

        if (threadIdx.x == 0) {
            done = 0;
            atomicExch(&status_queue[wid], PROC_READY);
            printf("[Worker %d] ready\n", wid);
        }
        __syncthreads();

        while (true) {
            if (threadIdx.x == 0) {
                while (atomicAdd(&doorbells[wid].ready, 0) == 0) {}
                int signal = atomicExch(&doorbells[wid].ready, 0);
                if (signal == 2) {
                    current.type = (TaskType)-1;
                } else {
                    int idx = doorbells[wid].task_idx;
                    current = tq->entry[idx & (DBG_CAPACITY - 1)];
                }
            }
            __syncthreads();

            if (current.type == (TaskType)-1) {
                if (threadIdx.x == 0)
                    printf("[Worker %d] EXIT after %d tasks\n", wid, done);
                break;
            }

            // "Execute" — just count
            if (threadIdx.x == 0) {
                done++;
                atomicAdd(task_count, 1);
                // Fan-in: decrement counter, if last push FFN2
                // (skip for now — just test scheduling)
                atomicExch(&status_queue[wid], PROC_READY);
            }
            __syncthreads();
        }
    }
}

int main() {
    TaskQueue<DBG_CAPACITY> *tq;
    Doorbell *doorbells;
    int *status_queue, *ffn1_done, *task_count;

    CudaAllocator::allocate(&tq, 1);
    CudaAllocator::allocate(&doorbells, DBG_WORKERS);
    CudaAllocator::allocate(&status_queue, DBG_WORKERS);
    CudaAllocator::allocate(&ffn1_done, constants::TOP_K);
    CudaAllocator::allocate(&task_count, 1);

    cudaMemset(tq, 0, sizeof(TaskQueue<DBG_CAPACITY>));
    cudaMemset(doorbells, 0, DBG_WORKERS * sizeof(Doorbell));
    cudaMemset(status_queue, 0, DBG_WORKERS * sizeof(int));
    cudaMemset(task_count, 0, sizeof(int));

    int ffn1_init[constants::TOP_K];
    for (int i = 0; i < constants::TOP_K; i++)
        ffn1_init[i] = constants::FFN1_TILES_PER_EXPERT;
    CudaAllocator::copy_to_device(ffn1_init, ffn1_done, constants::TOP_K);

    int expected = constants::TOP_K * constants::FFN1_TILES_PER_EXPERT;
    printf("Launching: 1 OS + %d workers, %d FFN1 tasks expected\n", DBG_WORKERS, expected);
    printf("CAPACITY=%d  TILE_ROWS=%d  FFN1_TILES_PER_EXPERT=%d\n\n",
           DBG_CAPACITY, constants::TILE_ROWS, constants::FFN1_TILES_PER_EXPERT);

    dbg_kernel<<<1 + DBG_WORKERS, 128>>>(tq, doorbells, status_queue, ffn1_done, task_count);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int h_count = 0;
    CudaAllocator::copy_to_host(task_count, &h_count, 1);
    printf("\nTotal tasks executed: %d/%d  %s\n", h_count, expected,
           h_count == expected ? "PASS" : "FAIL");

    CudaAllocator::free(tq);
    CudaAllocator::free(doorbells);
    CudaAllocator::free(status_queue);
    CudaAllocator::free(ffn1_done);
    CudaAllocator::free(task_count);

    return h_count == expected ? 0 : 1;
}
