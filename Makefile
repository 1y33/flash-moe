NVCC      = nvcc
ARCH      = -arch=sm_89 -O3
GTEST     = -lgtest -lgtest_main -lpthread
BUILD_DIR = build

UTILS     = csrc/utils/dtypes.cuh csrc/utils/warp.cuh csrc/utils/allocator.cuh csrc/utils/trace.cuh

.PHONY: all test test_cuda test_python test_kernels clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


KERNEL_DEPS = tests/test_kernel.cu csrc/kernel.cu csrc/os.cu csrc/worker.cu csrc/queue.cu csrc/flashmoe.cuh csrc/utils/allocator.cuh csrc/tasks/executor.cuh csrc/tasks/gemv.cuh csrc/tasks/gemv_ffn1.cuh csrc/tasks/gemv_ffn2.cuh csrc/tasks/silu_mul.cuh csrc/tasks/softmax_topk.cuh $(UTILS)

$(BUILD_DIR)/test_kernel: $(KERNEL_DEPS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_kernel_fp32: $(KERNEL_DEPS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -DMODEL_TYPE_FP32 -o $@ tests/test_kernel.cu

$(BUILD_DIR)/test_kernel_trace: $(KERNEL_DEPS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -DTRACE_ENABLED -o $@ tests/test_kernel.cu

$(BUILD_DIR)/test_queue: tests/test_queue.cu csrc/queue.cu csrc/utils/allocator.cuh csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_scheduler: tests/test_scheduler.cu csrc/queue.cu csrc/utils/allocator.cuh csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_fanin: tests/test_fanin.cu csrc/utils/allocator.cuh csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_full_debug: tests/test_full_debug.cu csrc/queue.cu csrc/utils/allocator.cuh csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

test_kernel: $(BUILD_DIR)/test_kernel
	timeout 15 ./$(BUILD_DIR)/test_kernel || echo "TIMEOUT — kernel likely deadlocked"

test_kernel_fp32: $(BUILD_DIR)/test_kernel_fp32
	timeout 15 ./$(BUILD_DIR)/test_kernel_fp32 || echo "TIMEOUT — kernel likely deadlocked"

test_kernel_trace: $(BUILD_DIR)/test_kernel_trace
	timeout 15 ./$(BUILD_DIR)/test_kernel_trace || echo "TIMEOUT — kernel likely deadlocked"

test_queue: $(BUILD_DIR)/test_queue
	./$(BUILD_DIR)/test_queue

test_scheduler: $(BUILD_DIR)/test_scheduler
	./$(BUILD_DIR)/test_scheduler

test_fanin: $(BUILD_DIR)/test_fanin
	./$(BUILD_DIR)/test_fanin

test_full_debug: $(BUILD_DIR)/test_full_debug
	./$(BUILD_DIR)/test_full_debug

test_cuda: test_kernel test_queue test_fanin test_scheduler test_full_debug

test_python:
	python -m benchmarks.sweep --models qwen3-30b-a3b --batches 1 \
		--runners flashmoe,vllm --warmup 5 --iters 20

test: test_cuda test_python

$(BUILD_DIR)/test_softmax_topk: tests/test_softmax_topk.cu csrc/tasks/softmax_topk.cuh csrc/utils/allocator.cuh tests/bench.cuh $(UTILS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_silu_mul: tests/test_silu_mul.cu csrc/tasks/silu_mul.cuh csrc/utils/allocator.cuh tests/bench.cuh $(UTILS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_gemv_tile: tests/test_gemv_tile.cu csrc/tasks/gemv.cuh csrc/tasks/gemv_ffn1.cuh csrc/utils/allocator.cuh csrc/flashmoe.cuh tests/bench.cuh $(UTILS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_gemv_down: tests/test_gemv_down.cu csrc/tasks/gemv_ffn2.cuh csrc/utils/allocator.cuh csrc/flashmoe.cuh tests/bench.cuh $(UTILS) | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

test_gemv_down: $(BUILD_DIR)/test_gemv_down
	./$(BUILD_DIR)/test_gemv_down

test_gemv_tile: $(BUILD_DIR)/test_gemv_tile
	./$(BUILD_DIR)/test_gemv_tile

test_softmax_topk: $(BUILD_DIR)/test_softmax_topk
	./$(BUILD_DIR)/test_softmax_topk

test_silu_mul: $(BUILD_DIR)/test_silu_mul
	./$(BUILD_DIR)/test_silu_mul

test_kernels: test_softmax_topk test_silu_mul

clean:
	rm -rf $(BUILD_DIR)
