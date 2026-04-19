// Test the fan-in mechanism: N threads atomicSub a counter,
// exactly one sees old==1 and becomes the "last arriver".
//
// Build:  make test_fanin
// Run:    ./build/test_fanin

#include <cstdio>
#include <cuda_runtime.h>
#include "../csrc/allocator.cu"

constexpr int N_PARENTS = 8;

__global__ void test_fanin_kernel(int *counter, int *last_arriver_count) {
    // Each thread is a "parent task" that finishes
    if (threadIdx.x < N_PARENTS) {
        __threadfence();
        int before = atomicSub(counter, 1);
        if (before == 1) {
            // I'm the last one
            atomicAdd(last_arriver_count, 1);
        }
    }
}

int main() {
    int pass = 0, fail = 0;

    // Run multiple times to catch races
    for (int trial = 0; trial < 100; trial++) {
        int *counter, *last_count;
        CudaAllocator::allocate(&counter, 1);
        CudaAllocator::allocate(&last_count, 1);

        int init = N_PARENTS;
        CudaAllocator::copy_to_device(&init, counter, 1);
        cudaMemset(last_count, 0, sizeof(int));

        test_fanin_kernel<<<1, N_PARENTS>>>(counter, last_count);
        cudaDeviceSynchronize();

        int h_counter, h_last;
        CudaAllocator::copy_to_host(counter, &h_counter, 1);
        CudaAllocator::copy_to_host(last_count, &h_last, 1);

        if (h_counter != 0 || h_last != 1) {
            printf("  FAIL trial %d: counter=%d last_arrivers=%d\n",
                   trial, h_counter, h_last);
            fail++;
        } else {
            pass++;
        }

        CudaAllocator::free(counter);
        CudaAllocator::free(last_count);
    }

    printf("[Fan-in] %d/100 passed\n", pass);
    return fail > 0 ? 1 : 0;
}
