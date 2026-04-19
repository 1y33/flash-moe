NVCC      = nvcc
ARCH      = -arch=sm_89
GTEST     = -lgtest -lgtest_main -lpthread
BUILD_DIR = build

.PHONY: all test clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_simple: tests/test_simple.cu csrc/queue.cu csrc/worker.cu csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $< $(GTEST)

$(BUILD_DIR)/gemv: playground/gemv.cu | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_gemv: playground/test_gemv.cu csrc/tasks/gemv.cuh csrc/allocator.cu playground/bench.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_softmax_topk: playground/test_softmax_topk.cu csrc/tasks/softmax_topk.cuh csrc/tasks/topk.cuh csrc/allocator.cu playground/bench.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_topk: playground/test_topk.cu csrc/tasks/topk.cuh csrc/allocator.cu playground/bench.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_silu_mul: playground/test_silu_mul.cu csrc/tasks/silu_mul.cuh csrc/allocator.cu playground/bench.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_ffn: playground/test_ffn.cu csrc/tasks/ffn.cuh csrc/tasks/gemv.cuh csrc/tasks/silu_mul.cuh csrc/allocator.cu playground/bench.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

gemv: $(BUILD_DIR)/gemv
	./$(BUILD_DIR)/gemv

test_gemv: $(BUILD_DIR)/test_gemv
	./$(BUILD_DIR)/test_gemv

test_softmax_topk: $(BUILD_DIR)/test_softmax_topk
	./$(BUILD_DIR)/test_softmax_topk

test_topk: $(BUILD_DIR)/test_topk
	./$(BUILD_DIR)/test_topk

test_silu_mul: $(BUILD_DIR)/test_silu_mul
	./$(BUILD_DIR)/test_silu_mul

test_ffn: $(BUILD_DIR)/test_ffn
	./$(BUILD_DIR)/test_ffn

$(BUILD_DIR)/test_queue: tests/test_queue.cu csrc/queue.cu csrc/allocator.cu csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_scheduler: tests/test_scheduler.cu csrc/queue.cu csrc/allocator.cu csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

$(BUILD_DIR)/test_fanin: tests/test_fanin.cu csrc/allocator.cu csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

test_queue: $(BUILD_DIR)/test_queue
	./$(BUILD_DIR)/test_queue

test_scheduler: $(BUILD_DIR)/test_scheduler
	./$(BUILD_DIR)/test_scheduler

test_fanin: $(BUILD_DIR)/test_fanin
	./$(BUILD_DIR)/test_fanin

test_kernels: test_gemv test_topk test_softmax_topk test_silu_mul test_ffn

$(BUILD_DIR)/test_full_debug: tests/test_full_debug.cu csrc/queue.cu csrc/allocator.cu csrc/flashmoe.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

test_full_debug: $(BUILD_DIR)/test_full_debug
	./$(BUILD_DIR)/test_full_debug

$(BUILD_DIR)/test_kernel: tests/test_kernel.cu csrc/os.cu csrc/worker.cu csrc/queue.cu csrc/flashmoe.cuh csrc/allocator.cu csrc/tasks/executor.cuh csrc/tasks/gemv.cuh csrc/tasks/silu_mul.cuh csrc/tasks/softmax_topk.cuh csrc/tasks/topk.cuh | $(BUILD_DIR)
	$(NVCC) $(ARCH) -o $@ $<

test_kernel: $(BUILD_DIR)/test_kernel
	timeout 15 ./$(BUILD_DIR)/test_kernel || echo "TIMEOUT — kernel likely deadlocked"

test_system: test_queue test_fanin test_scheduler

test_all: test_kernels test_system

clean:
	rm -rf $(BUILD_DIR)
