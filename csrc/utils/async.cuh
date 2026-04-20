#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct CopyAsync {
    __device__ __forceinline__ static void copy16(void *dst, const void *src) {
        uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(dst_addr), "l"(src));
    }

    __device__ __forceinline__ static void commit() {
        asm volatile("cp.async.commit_group;\n");
    }

    // Wait until at most N groups are still in flight
    // wait<0>() = wait for everything
    // wait<1>() = wait until only 1 group remains (the latest)
    template <int N>
    __device__ __forceinline__ static void wait() {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    }

    // Wait for all groups to complete
    __device__ __forceinline__ static void wait_all() {
        asm volatile("cp.async.wait_all;\n");
    }
};
