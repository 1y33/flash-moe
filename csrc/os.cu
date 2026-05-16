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

    // Warp-parallel dispatch. Total tasks = TOP_K * (FFN1_TILES + FFN2_TILES).
    // FFN1 tasks come first (indices [0, TOP_K*FFN1_TILES_PER_EXPERT)), then FFN2.
    // All 32 lanes of warp 0 push in parallel; each does ~6 round-trips.
    // Each task's queue slot is determined by its global task index, so we
    // don't need a contended atomicAdd on head — we set head = total at the end.
    static __device__ __forceinline__ void dispatch(TaskQueue<constants::CAPACITY> *task_queue,
                                                    int *expert_ids, float *expert_weights)
    {
        constexpr int FFN1_TASKS = constants::TOP_K * constants::FFN1_TILES_PER_EXPERT;
        constexpr int FFN2_TASKS = constants::TOP_K * constants::FFN2_TILES_PER_EXPERT;
        constexpr int TOTAL = FFN1_TASKS + FFN2_TASKS;

        const int lane = threadIdx.x & 31;

        for (int i = lane; i < TOTAL; i += 32)
        {
            Task t;
            if (i < FFN1_TASKS)
            {
                int k = i / constants::FFN1_TILES_PER_EXPERT;
                int tile = i % constants::FFN1_TILES_PER_EXPERT;
                t.expert_id = expert_ids[k];
                t.weight = expert_weights[k];
                t.type = FFN1;
                t.row_begin = tile * constants::TILE_ROWS;
                int rows_left = constants::MOE_INTERMEDIATE_SIZE - t.row_begin;
                t.row_count = rows_left < constants::TILE_ROWS ? rows_left : constants::TILE_ROWS;
                t.slot = k;
            }
            else
            {
                int j = i - FFN1_TASKS;
                int k = j / constants::FFN2_TILES_PER_EXPERT;
                int tile = j % constants::FFN2_TILES_PER_EXPERT;
                t.expert_id = expert_ids[k];
                t.weight = expert_weights[k];
                t.type = FFN2;
                t.row_begin = tile * constants::TILE_ROWS;
                int rows_left = constants::HIDDEN_SIZE - t.row_begin;
                t.row_count = rows_left < constants::TILE_ROWS ? rows_left : constants::TILE_ROWS;
                t.slot = k;
            }

            int slot = i & (constants::CAPACITY - 1);
            task_queue->entry[slot] = t;
            __threadfence();
            atomicExch(&task_queue->slot_ready[slot], 1);
        }

        __syncwarp();
        if (lane == 0)
        {
            // Make all tasks visible to the scheduler. head is read with
            // atomicAdd(0) in pop(), so a normal store with threadfence is fine.
            __threadfence();
            atomicExch(&task_queue->head, TOTAL);
        }
    }
};

// Worker self-dispatch: no Scheduler needed. The OS block just does topk +
// warp-parallel dispatch and exits. Workers (blocks 1..BLOCKSIZE-1) pull tasks
// from the queue directly via atomicAdd(&tail, 1).
template <typename T>
struct OS
{
    static __device__ __forceinline__ void run(float *logits, FlashMoe<T> *model,
                                               TaskQueue<constants::CAPACITY> *task_queue,
                                               DeviceTracer tracer, long long *pending)
    {
        __shared__ int expert_ids[constants::TOP_K];
        __shared__ float expert_weights[constants::TOP_K];

        bool is_lane0 = (threadIdx.x % 32 == 0);

        if (is_lane0)
            tracer.start(TR_ROUTE, pending);

        int warp_id = threadIdx.x / 32;

        if (warp_id == 0)
        {
            if (threadIdx.x == 0)
                tracer.start(TR_SOFTMAX_TOPK, pending);
            BootStrap<T>::topk(logits, expert_ids, expert_weights);
            __syncwarp();
            if (threadIdx.x == 0)
                tracer.stop(TR_SOFTMAX_TOPK, pending);

            if (threadIdx.x == 0)
                tracer.start(TR_DISPATCH, pending);
            BootStrap<T>::dispatch(task_queue, expert_ids, expert_weights);
            if (threadIdx.x == 0)
            {
                tracer.stop(TR_DISPATCH, pending);
                tracer.stop(TR_ROUTE, pending);
            }
        }
        // No scheduler — OS block exits after dispatch.
    }
};
