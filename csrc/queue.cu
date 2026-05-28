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
    int slot_ready[capacity]; // 0=empty, 1=written ... per-slot visibility flag
    int head;
    int tail;

    __device__ __forceinline__ bool push(Task t)
    {
        int idx = atomicAdd(&head, 1);
        int slot = idx & (capacity - 1);
        entry[slot] = t;

        __threadfence();
        atomicExch(&slot_ready[slot], 1);

        return true;
    }

    __device__ __forceinline__ bool pop(int *out_idx)
    {
        int h = atomicAdd(&head, 0);
        if (tail >= h)
            return false;

        int slot = tail & (capacity - 1);
        while (atomicAdd(&slot_ready[slot], 0) == 0)
        {
        } // wait for producer

        atomicExch(&slot_ready[slot], 0);
        *out_idx = tail;
        tail++;
        return true;
    }
};
