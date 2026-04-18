# FlashMoE

Single-kernel Mixture of Experts operator for NVIDIA GPUs, inspired by the [FlashDMoE paper](https://arxiv.org/abs/2506.04667).

Targets **single-token decode** through one MoE layer (Qwen3-30B-A3B config) on RTX 4070.

## Architecture

A persistent kernel with an embedded OS:

- **OS block** (1 SM) — bootstrap (router GEMV + softmax top-K) + scheduler (assigns tasks to workers via doorbells)
- **Worker blocks** (45 SMs) — poll doorbells, execute FFN tiles, push follow-up tasks on fan-in completion

```
Bootstrap → push FFN1 tiles → Scheduler → doorbells → Workers
                                    ↑                      │
                                    └── FFN2 tiles ←───────┘
```

## Config

Hardcoded in `csrc/flashmoe.cuh` to match Qwen3-30B-A3B:

| Parameter | Value |
|-----------|-------|
| HIDDEN_SIZE | 2048 |
| MOE_INTERMEDIATE_SIZE | 768 |
| NUM_EXPERTS | 128 |
| TOP_K | 8 |
| Activation | SiLU (SwiGLU) |

## Build & Test

Requires: CUDA toolkit, PyTorch with CUDA support.

```bash
# Standalone kernel tests (no Python)
make test_kernels

# Smoke test (JIT compiles, no vLLM)
.venv/bin/python -m benchmarks.test_moe

# Correctness test vs vLLM
.venv/bin/python -m benchmarks.test_correctness

# vLLM-only benchmark
.venv/bin/python -m benchmarks.bench_vllm
```

## Project Structure

```
csrc/
├── flashmoe.cuh          # constants + FlashMoe/ExpertFFN structs
├── allocator.cu           # CudaAllocator / HostAllocator
├── queue.cu               # Task, Doorbell, TaskQueue, ProcStatus
├── os.cu                  # BootStrap, Scheduler, OS
├── worker.cu              # Worker (doorbell loop + task router)
├── kernel.cu              # __global__ kernel + host launch
├── binding.cu             # PyTorch/pybind11 bindings
└── tasks/
    ├── gemv.cuh           # GEMV primitives (tile + accumulate)
    ├── silu_mul.cuh       # SiLU activation
    ├── topk.cuh           # warp-level top-K
    ├── softmax_topk.cuh   # fused softmax + top-K
    ├── ffn.cuh            # FFN1/FFN2 composites
    └── executor.cuh       # FFN1Executor / FFN2Executor

benchmarks/
├── vllm_utils.py          # shared vLLM setup helpers
├── bench_vllm.py          # vLLM benchmark
├── test_correctness.py    # our kernel vs vLLM comparison
└── test_moe.py            # standalone smoke test

notes/
├── scheduling.md          # design doc (actor model)
└── doorbell_system.md     # how the doorbell dispatch works
```
