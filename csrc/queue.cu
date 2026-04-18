#pragma once
#include <cuda_runtime.h>

enum TaskType : int
{
    FFN1,
    FFN2
};

struct Task
{
    int expert_id;
    float weight;
    TaskType type;
    int row_begin;
    int row_count;
    int slot;
};
struct Doorbell
{
    int task_idx;
    int ready;
};

enum ProcStatus : int
{
    PROC_READY = 0,
    PROC_BUSY = 1
};

template <int capacity>
struct TaskQueue
{
    Task entry[capacity];
    int head; 
    int tail; 

    __device__ __forceinline__ bool push(Task t)
    {
        int idx = atomicAdd(&head, 1);
        entry[idx & (capacity - 1)] = t;
        __threadfence();
        return true;
    }

    __device__ __forceinline__ bool pop(int *out_idx)
    {
        int h = atomicAdd(&head, 0);
        if (tail >= h)
            return false;
        *out_idx = tail;
        tail++;
        return true;
    }
};

