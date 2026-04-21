#include <cuda_runtime.h>
#include "tasks/gemv.cuh"
#include "tasks/softmax_topk.cuh"
#include "flashmoe.cuh"
#include "queue.cu"

template <typename T>
struct BootStrap
{
    static __device__ __forceinline__ void route(T *input, float *logits,
                                                 int *expert_ids, float *expert_weights,
                                                 FlashMoe<T> *model)
    {
        flashmoe::gemv_tile<T, constants::THREADS_PER_BLOCK>(
            model->router, input, logits,
            constants::HIDDEN_SIZE, 0, constants::NUM_EXPERTS);
    }

    static __device__ __forceinline__ void topk(float *logits, int *expert_ids, float *expert_weights)
    {
        flashmoe::softmax_topk_warp<constants::NUM_EXPERTS, constants::TOP_K>(
            logits, expert_ids, expert_weights);
    }

    static __device__ __forceinline__ void dispatch(TaskQueue<constants::CAPACITY> *task_queue,
                                                    int *expert_ids, float *expert_weights)
    {
        if (threadIdx.x == 0)
        {
            for (int k = 0; k < constants::TOP_K; k++)
            {
                int rows_left = constants::MOE_INTERMEDIATE_SIZE;
                int row = 0;
                while (rows_left > 0)
                {
                    int chunk = min(constants::TILE_ROWS, rows_left);
                    Task t;
                    t.expert_id = expert_ids[k];
                    t.weight = expert_weights[k];
                    t.type = FFN1;
                    t.row_begin = row;
                    t.row_count = chunk;
                    t.slot = k;
                    task_queue->push(t);
                    row += chunk;
                    rows_left -= chunk;
                }
            }
        }
    }
};

struct Scheduler
{
    static __device__ __forceinline__ void assign_task(int worker_id, int task_idx,
                                                       Doorbell *doorbells)
    {
        doorbells[worker_id].task_idx = task_idx;
        __threadfence();
        atomicExch(&doorbells[worker_id].ready, 1);
    }

    static __device__ __forceinline__ void send_exit(int worker_id, int *status_queue,
                                                     Doorbell *doorbells)
    {
        while (atomicExch(&status_queue[worker_id], PROC_BUSY) != PROC_READY)
        {
        }
        atomicExch(&doorbells[worker_id].ready, 2);
    }

    static __device__ __forceinline__ int find_ready_worker(int *status_queue, int num_workers, int &next_w)
    {
        while (true)
        {
            for (int attempt = 0; attempt < num_workers; attempt++)
            {
                int w = next_w;
                next_w = (next_w + 1) % num_workers;
                int old = atomicExch(&status_queue[w], PROC_BUSY);
                if (old == PROC_READY)
                    return w;
            }
        }
    }

    static __device__ __forceinline__ void run(TaskQueue<constants::CAPACITY> *task_queue,
                                               Doorbell *doorbells,
                                               int *status_queue,
                                               int num_workers,
                                               int total_tasks,
                                               DeviceTracer &tracer, long long *pending)
    {
        int scheduled = 0;
        int next_w = 0;

        tracer.start(TR_SCHEDULE, pending);
        while (scheduled < total_tasks)
        {
            int task_idx;
            if (!task_queue->pop(&task_idx))
                continue;

            int w = find_ready_worker(status_queue, num_workers, next_w);
            assign_task(w, task_idx, doorbells);
            scheduled++;
        }
        tracer.stop(TR_SCHEDULE, pending);

        for (int w = 0; w < num_workers; w++)
        {
            send_exit(w, status_queue, doorbells);
        }
    }
};

template <typename T>
struct OS
{
    static __device__ __forceinline__ void run(T *input, FlashMoe<T> *model,
                                               TaskQueue<constants::CAPACITY> *task_queue,
                                               Doorbell *doorbells,
                                               int *status_queue,
                                               int *ffn1_done,
                                               int num_workers,
                                               int total_tasks,
                                               DeviceTracer tracer, long long *pending)
    {
        __shared__ float logits[constants::NUM_EXPERTS];
        __shared__ int expert_ids[constants::TOP_K];
        __shared__ float expert_weights[constants::TOP_K];
        __shared__ int bootstrap_done;

        bool is_lane0 = (threadIdx.x % 32 == 0);

        if (threadIdx.x == 0)
            bootstrap_done = 0;
        __syncthreads();

        // Outer ROUTE group span
        if (is_lane0) tracer.start(TR_ROUTE, pending);

        // Inner: GEMV_ROUTE
        if (is_lane0) tracer.start(TR_GEMV_ROUTE, pending);
        BootStrap<T>::route(input, logits, expert_ids, expert_weights, model);
        __syncthreads();
        if (is_lane0) tracer.stop(TR_GEMV_ROUTE, pending);

        int warp_id = threadIdx.x / 32;

        if (warp_id == 0)
        {
            // Inner: SOFTMAX_TOPK
            if (threadIdx.x == 0) tracer.start(TR_SOFTMAX_TOPK, pending);
            BootStrap<T>::topk(logits, expert_ids, expert_weights);
            __syncwarp();
            if (threadIdx.x == 0) tracer.stop(TR_SOFTMAX_TOPK, pending);

            // Inner: DISPATCH
            if (threadIdx.x == 0) tracer.start(TR_DISPATCH, pending);
            BootStrap<T>::dispatch(task_queue, expert_ids, expert_weights);
            if (threadIdx.x == 0)
            {
                tracer.stop(TR_DISPATCH, pending);
                // Close outer ROUTE group
                tracer.stop(TR_ROUTE, pending);
                __threadfence();
                atomicExch(&bootstrap_done, 1);
            }
        }
        else if (warp_id == 1 && threadIdx.x == 32)
        {
            while (atomicAdd(&bootstrap_done, 0) == 0) { }
            Scheduler::run(task_queue, doorbells, status_queue,
                           num_workers, total_tasks, tracer, pending);
        }
    }
};
