#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "../csrc/flashmoe.cuh"
#include "../csrc/worker.cu"

using namespace constants;

constexpr int NUM_TASKS = 32;

// concrete worker: adds start + finish, writes to results[type]
struct AddWorker : WorkerBase<AddWorker, CAPACITY>{
    int *results;
    int *pop_count;

    __device__ __forceinline__ void execute(Task *t){
        results[t->type] = t->start + t->finish;
        atomicAdd(pop_count, 1);
    }
};

// block 0 = producer, block 1 = consumer using AddWorker
__global__ void test_add_worker_kernel(TaskQueue<CAPACITY> *q, int *results, int *pop_count)
{
    Scheduler<CAPACITY> sched;

    if (is_scheduler() == 0)
    {
        if (threadIdx.x < NUM_TASKS)
        {
            Task t;
            t.type   = threadIdx.x;
            t.start  = threadIdx.x;
            t.finish = threadIdx.x * 10;

            sched.push_task(q, t);
        }
    }
    else
    {
        __nanosleep(1000000);

        AddWorker worker;
        worker.results   = results;
        worker.pop_count = pop_count;
        worker.run(sched, q);
    }
}

// helper: alloc zeroed device memory
template <typename T>
T *device_alloc(size_t count = 1)
{
    T *ptr;
    cudaMalloc(&ptr, sizeof(T) * count);
    cudaMemset(ptr, 0, sizeof(T) * count);
    return ptr;
}

class TaskQueueTest : public ::testing::Test
{
protected:
    TaskQueue<CAPACITY> *d_queue;
    int *d_results;
    int *d_pop_count;

    void SetUp() override
    {
        d_queue     = device_alloc<TaskQueue<CAPACITY>>();
        d_results   = device_alloc<int>(NUM_TASKS);
        d_pop_count = device_alloc<int>();
    }

    void TearDown() override
    {
        cudaFree(d_queue);
        cudaFree(d_results);
        cudaFree(d_pop_count);
    }
};

TEST_F(TaskQueueTest, AddWorkerPushPopAll)
{
    test_add_worker_kernel<<<2, NUM_TASKS>>>(d_queue, d_results, d_pop_count);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    int h_results[NUM_TASKS];
    int h_pop_count = 0;
    cudaMemcpy(h_results, d_results, NUM_TASKS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pop_count, d_pop_count, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_pop_count, NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; i++)
    {
        int expected = i + i * 10;
        EXPECT_EQ(h_results[i], expected) << "task " << i;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
