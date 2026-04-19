#include <cuda_runtime.h>
#include "tasks/gemv.cuh"
#include "tasks/softmax_topk.cuh"
#include "flashmoe.cuh"
#include "queue.cu"

struct BootStrap {
    static __device__ __forceinline__
    void route(float *input, float *logits,
               int *expert_ids, float *expert_weights,
               FlashMoe<float> *model)
    {
        // gemv_tile uses all threads in the block (not just warp 0)
        flashmoe::gemv_tile<128>(
            model->router, input, logits,
            constants::HIDDEN_SIZE, 0, constants::NUM_EXPERTS);
    }

    static __device__ __forceinline__
    void topk(float *logits, int *expert_ids, float *expert_weights)
    {
        // softmax_topk runs on warp 0 only — no syncthreads needed
        flashmoe::softmax_topk_warp<constants::NUM_EXPERTS, constants::TOP_K>(
            logits, expert_ids, expert_weights);
    }

    static __device__ __forceinline__
    void dispatch(TaskQueue<constants::CAPACITY> *task_queue,
                  int *expert_ids, float *expert_weights)
    {
        if (threadIdx.x == 0) {
            for (int k = 0; k < constants::TOP_K; k++) {
                int rows_left = constants::MOE_INTERMEDIATE_SIZE;
                int row = 0;
                while (rows_left > 0) {
                    int chunk = min(constants::TILE_ROWS, rows_left);
                    Task t;
                    t.expert_id = expert_ids[k];
                    t.weight    = expert_weights[k];
                    t.type      = FFN1;
                    t.row_begin = row;
                    t.row_count = chunk;
                    t.slot      = k;
                    task_queue->push(t);
                    row       += chunk;
                    rows_left -= chunk;
                }
            }
        }
    }
};

struct Scheduler {

    static __device__ __forceinline__
    int find_ready_worker(int *status_queue, int num_workers) {
        while (true) {
            for (int w = 0; w < num_workers; w++) {
                int old = atomicExch(&status_queue[w], PROC_BUSY);
                if (old == PROC_READY)
                    return w;
            }
        }
    }

    static __device__ __forceinline__
    void assign_task(int worker_id, int task_idx,
                     Doorbell *doorbells) {
        doorbells[worker_id].task_idx = task_idx;
        __threadfence();
        atomicExch(&doorbells[worker_id].ready, 1);
    }

    static __device__ __forceinline__
    void send_exit(int worker_id, int *status_queue,
                   Doorbell *doorbells) {
        while (atomicExch(&status_queue[worker_id], PROC_BUSY) != PROC_READY) {}
        atomicExch(&doorbells[worker_id].ready, 2);
    }

    static __device__ __forceinline__
    void run(TaskQueue<constants::CAPACITY> *task_queue,
             Doorbell *doorbells,
             int *status_queue,
             int num_workers,
             int total_tasks)
    {
        int scheduled = 0;

        while (scheduled < total_tasks) {
            int task_idx;
            if (!task_queue->pop(&task_idx))
                continue;

            int w = find_ready_worker(status_queue, num_workers);
            assign_task(w, task_idx, doorbells);
            scheduled++;
        }

        for (int w = 0; w < num_workers; w++) {
            send_exit(w, status_queue, doorbells);
        }
    }
};

struct OS {
    static __device__ __forceinline__
    void run(float *input, FlashMoe<float> *model,
             TaskQueue<constants::CAPACITY> *task_queue,
             Doorbell *doorbells,
             int *status_queue,
             int *ffn1_done,
             int num_workers,
             int total_tasks)
    {
        __shared__ float logits[constants::NUM_EXPERTS];
        __shared__ int   expert_ids[constants::TOP_K];
        __shared__ float expert_weights[constants::TOP_K];
        __shared__ int   bootstrap_done;

        if (threadIdx.x == 0)
            bootstrap_done = 0;

        // Phase 1: ALL threads do the router GEMV (needs full block)
        BootStrap::route(input, logits, expert_ids, expert_weights, model);
        __syncthreads();  // safe: all threads participate

        // Phase 2: warp 0 does topk + dispatch, warp 1 waits then schedules
        int warp_id = threadIdx.x / 32;

        if (warp_id == 0) {
            BootStrap::topk(logits, expert_ids, expert_weights);
            __syncwarp();  // ensure topk results visible within warp 0
            BootStrap::dispatch(task_queue, expert_ids, expert_weights);
            if (threadIdx.x == 0) {
                __threadfence();
                atomicExch(&bootstrap_done, 1);
            }
        }
        else if (warp_id == 1 && threadIdx.x == 32) {
            // Wait for bootstrap to finish pushing tasks
            while (atomicAdd(&bootstrap_done, 0) == 0) {}
            Scheduler::run(task_queue, doorbells, status_queue,
                           num_workers, total_tasks);
        }
    }
};
