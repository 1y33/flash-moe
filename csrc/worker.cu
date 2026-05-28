#pragma once
#include "queue.cu"
#include "flashmoe.cuh"
#include "tasks/executor.cuh"

template <typename T, typename AccT = __half>
struct Worker
{
    static __device__ __forceinline__ void route_task(Task &task, FlashMoe<T> *model,
                                                      T *input,
                                                      AccT *ffn1_out,
                                                      float *output,
                                                      int *ffn1_done,
                                                      DeviceTracer &tracer, long long *pending)
    {
        bool is_lane0 = (threadIdx.x % 32 == 0);

        switch (task.type)
        {
        case FFN1:
            if (is_lane0)
                tracer.start(TR_FFN1, pending);
            FFN1Executor<T, AccT>::execute(task, model, input, ffn1_out, tracer, pending);
            __threadfence();
            if (is_lane0)
                tracer.stop(TR_FFN1, pending);
            if (threadIdx.x == 0)
            {
                FFN1Executor<T, AccT>::on_complete(task, ffn1_done);
            }
            break;

        case FFN2:
            if (is_lane0)
                tracer.start(TR_FFN2, pending);
            FFN2Executor<T, AccT>::execute(task, model, ffn1_out, output, ffn1_done, tracer, pending);
            __threadfence();
            if (is_lane0)
                tracer.stop(TR_FFN2, pending);
            if (threadIdx.x == 0)
            {
                FFN2Executor<T, AccT>::on_complete(task);
            }
            break;
        }
    }

    // Self-dispatch: workers claim tasks from the queue directly.
    static __device__ __forceinline__ void run(int worker_id, FlashMoe<T> *model,
                                               T *input, AccT *ffn1_out, float *output,
                                               TaskQueue<constants::CAPACITY> *task_queue,
                                               int *ffn1_done,
                                               DeviceTracer tracer, long long *pending)
    {
        __shared__ Task current_task;
        __shared__ int my_idx;

        const int total_tasks = constants::TOTAL_TASKS;

        while (true)
        {
            if (threadIdx.x == 0)
            {
                tracer.start(TR_WAIT, pending);
                
                int idx = atomicAdd(&task_queue->tail, 1);
                my_idx = idx;
                if (idx < total_tasks)
                {
                    int slot = idx & (constants::CAPACITY - 1);
                    while (atomicAdd(&task_queue->slot_ready[slot], 0) == 0)
                    {
                    }
                    current_task = task_queue->entry[slot];
                }
                tracer.stop(TR_WAIT, pending);
            }
            __syncthreads();

            if (my_idx >= total_tasks)
                break;

            route_task(current_task, model, input, ffn1_out, output, ffn1_done, tracer, pending);
            __syncthreads();
        }
    }
};
