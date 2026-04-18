NVCC      = nvcc
ARCH      = -arch=sm_70
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

test_kernels: test_gemv test_topk test_softmax_topk test_silu_mul test_ffn

test: $(BUILD_DIR)/test_simple
	./$(BUILD_DIR)/test_simple

clean:
	rm -rf $(BUILD_DIR)
