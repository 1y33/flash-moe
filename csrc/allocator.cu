#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include "flashmoe.cuh"

struct CudaAllocator{
    template <typename T>
    static void allocate(T **ptr, size_t count){
        cudaMalloc(ptr, count * sizeof(T));
    }

    template <typename T>
    static void copy_to_device(const T *host, T *device, size_t count){
        cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    static void copy_to_host(const T *device, T *host, size_t count){
        cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <typename T>
    static void free(T *ptr){
        cudaFree(ptr);
    }
};

struct HostAllocator{
    template <typename T>
    static void allocate(T **ptr, size_t count){
        *ptr = (T*)std::malloc(count * sizeof(T));
    }

    template <typename T>
    static void fill_random(T *ptr, size_t count){
        for (size_t i = 0; i < count; i++)
            ptr[i] = (T)rand() / (T)RAND_MAX - (T)0.5;
    }

    template <typename T>
    static void free(T *ptr){
        std::free(ptr);
    }
};

template <typename T, typename Alloc = CudaAllocator>
void allocate_flashmoe(FlashMoe<T> &model){
    namespace C = constants;

    Alloc::template allocate<T>(&model.router, C::ROUTER_SIZE);

    for(int i = 0; i < C::NUM_EXPERTS; i++){
        Alloc::template allocate<T>(&model.experts[i].gate_proj, C::GATE_PROJ_SIZE);
        Alloc::template allocate<T>(&model.experts[i].up_proj, C::UP_PROJ_SIZE);
        Alloc::template allocate<T>(&model.experts[i].down_proj, C::DOWN_PROJ_SIZE);
    }
}

template <typename T, typename Alloc = CudaAllocator>
void free_flashmoe(FlashMoe<T> &model){
    Alloc::template free<T>(model.router);

    for(int i = 0; i < constants::NUM_EXPERTS; i++){
        Alloc::template free<T>(model.experts[i].gate_proj);
        Alloc::template free<T>(model.experts[i].up_proj);
        Alloc::template free<T>(model.experts[i].down_proj);
    }
}
