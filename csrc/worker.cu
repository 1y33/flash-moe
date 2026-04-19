#pragma once
#include "queue.cu"
#include "flashmoe.cuh"
#include "tasks/executor.cuh"

template <typename T>
struct Worker
{
    static __device__ __forceinline__ int wait_for_doorbell(int worker_id, Doorbell *doorbells)
    {
        while (atomicAdd(&doorbells[worker_id].ready, 0) == 0)
        {
        }
        return atomicExch(&doorbells[worker_id].ready, 0);
    }

    static __device__ __forceinline__
        Task
        load_task(int worker_id, Doorbell *doorbells,
                  TaskQueue<constants::CAPACITY> *task_queue)
    {
        int idx = doorbells[worker_id].task_idx;
        return task_queue->entry[idx & (constants::CAPACITY - 1)];
    }

    static __device__ __forceinline__ void mark_ready(int worker_id, int *status_queue)
    {
        atomicExch(&status_queue[worker_id], PROC_READY);
    }

    static __device__ __forceinline__ void route_task(Task &task, FlashMoe<T> *model,
                                                      T *input, float *ffn1_out, float *output,
                                                      int *ffn1_done, TaskQueue<constants::CAPACITY> *task_queue)
    {
        switch (task.type)
        {
   
            case FFN1:
            FFN1Executor<T>::execute(task, model, input, ffn1_out);
            __threadfence();
            if (threadIdx.x == 0 && FFN1Executor<T>::on_complete(task, ffn1_done))
            {
                FFN1Executor<T>::push_next(task, task_queue);
            }
            break;

        case FFN2:
        
            FFN2Executor<T>::execute(task, model, ffn1_out, output);
            __threadfence();
            if (threadIdx.x == 0)
            {
                FFN2Executor<T>::on_complete(task);
            }
            break;
        }
    }

    static __device__ __forceinline__ void run(int worker_id, FlashMoe<T> *model,
                                               T *input, float *ffn1_out, float *output,
                                               TaskQueue<constants::CAPACITY> *task_queue,
                                               Doorbell *doorbells, int *status_queue, int *ffn1_done)
    {
        __shared__ Task current_task;

        if (threadIdx.x == 0)
            mark_ready(worker_id, status_queue);

        while (true)
        {
            if (threadIdx.x == 0)
            {
                int signal = wait_for_doorbell(worker_id, doorbells);
                if (signal == 2)
                {
                    current_task.type = (TaskType)-1;
                }
                else
                {
                    current_task = load_task(worker_id, doorbells, task_queue);
                }
            }
            __syncthreads();

            if (current_task.type == (TaskType)-1)
                break;

            route_task(current_task, model, input, ffn1_out, output, ffn1_done, task_queue);
            __syncthreads();

            if (threadIdx.x == 0)
                mark_ready(worker_id, status_queue);
        }
    }
};
