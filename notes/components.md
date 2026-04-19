# Component Reference

## File Map

```
csrc/
├── flashmoe.cuh      Constants + model structs
├── allocator.cu       Memory allocation helpers
├── queue.cu           Task, Doorbell, TaskQueue, ProcStatus
├── os.cu              BootStrap, Scheduler, OS
├── worker.cu          Worker (doorbell loop + router)
├── kernel.cu          __global__ kernel + host launch function
├── binding.cu         PyTorch/pybind11 bindings
└── tasks/
    ├── gemv.cuh       GEMV primitives
    ├── silu_mul.cuh   SiLU activation
    ├── topk.cuh       Warp-level top-K
    ├── softmax_topk.cuh  Fused softmax + top-K
    ├── ffn.cuh        FFN1/FFN2 composites (for standalone tests)
    └── executor.cuh   FFN1Executor / FFN2Executor (used by workers)
```

---

## flashmoe.cuh — The Constants

All model dimensions and derived values live here. Change these to
target a different model.

```cuda
namespace constants {
    BLOCKSIZE = 46           // total SM count on RTX 4070
    CAPACITY = 512           // task queue ring buffer size
    HIDDEN_SIZE = 2048       // H
    MOE_INTERMEDIATE_SIZE = 768  // I
    NUM_EXPERTS = 128        // E
    TOP_K = 8                // K

    TILE_ROWS = 96           // rows per tile/task
    FFN1_TILES_PER_EXPERT = I / TILE_ROWS           = 8
    FFN2_TILES_PER_EXPERT = ceil(H / TILE_ROWS)     = 22
    NUM_WORKERS = BLOCKSIZE - 1                      = 45
    TOTAL_TASKS = K * (FFN1_TILES + FFN2_TILES)      = 240
}
```

`FlashMoe<T>` holds pointers to all weight matrices:
```
model.router                        → [E, H]
model.experts[e].gate_proj          → [I, H]
model.experts[e].up_proj            → [I, H]
model.experts[e].down_proj          → [H, I]
```

---

## queue.cu — Scheduling Primitives

### Task
```cuda
struct Task {
    int expert_id;    // which expert
    float weight;     // softmax'd router weight
    TaskType type;    // FFN1 or FFN2
    int row_begin;    // starting row of this tile
    int row_count;    // number of rows in this tile
    int slot;         // 0..K-1, identifies which top-K slot
};
```
This is the "message" that flows through the system. Small and fixed-size.

### TaskQueue<capacity>
Ring buffer. Multiple producers push (atomicAdd on head), single
consumer pops (scheduler reads tail, no contention).

- `push(Task t)` → atomicAdd head, write entry, threadfence
- `pop(int *out_idx)` → check head vs tail, return index

### Doorbell
Per-worker mailbox. Scheduler writes task index + rings bell.
Worker polls until bell rings.
```cuda
struct Doorbell {
    int task_idx;   // index into TaskQueue
    int ready;      // 0=idle, 1=task, 2=EXIT
};
```

### ProcStatus
Worker state: `PROC_READY=0` or `PROC_BUSY=1`.
Worker sets READY when idle. Scheduler atomicExch's to BUSY when claiming.

---

## os.cu — The OS Block

Runs on block 0. Three phases:

### Phase 1: Route (all threads)
`gemv_tile`: router[E,H] @ input[H] → logits[E]
Uses all 128 threads (4 warps). __syncthreads after.

### Phase 2: TopK + Dispatch (warp 0 only)
`softmax_topk_warp`: logits → expert_ids[K], weights[K]
Then pushes FFN1 tile tasks into TaskQueue.
Sets `bootstrap_done` flag via atomicExch.

### Phase 3: Scheduler (warp 1, thread 32 only)
Waits for `bootstrap_done`. Then loops:
1. `pop()` a task index from the queue
2. `find_ready_worker()` — scan status_queue with atomicExch
3. `assign_task()` — write doorbell, threadfence, ring bell
4. Repeat until `scheduled == TOTAL_TASKS`
5. `send_exit()` to all workers

---

## worker.cu — The Worker Block

Runs on blocks 1..45. Each is an independent actor.

### Loop:
```
mark_ready()
while true:
    wait_for_doorbell()   → thread 0 spins on doorbell
    load_task()           → read Task from queue by index
    broadcast via __shared__
    route_task()          → switch(type) → FFN1Executor or FFN2Executor
    mark_ready()
```

### route_task dispatches to:
- **FFN1**: execute → threadfence → fan-in check → maybe push FFN2 tasks
- **FFN2**: execute → threadfence → on_complete

---

## tasks/executor.cuh — FFN Executors

### FFN1Executor::execute
For a tile of `row_count` rows starting at `row_begin`:
1. `gemv_tile`: gate_proj[row_begin:+row_count, :] @ input → act (global mem)
2. `gemv_tile`: up_proj[row_begin:+row_count, :] @ input → up_smem (shared mem)
3. `silu_mul`: act[r] = silu(gate[r]) * up[r]

Why shared memory for up? Multiple workers run concurrently on different
tiles. Global scratch would collide. Shared memory is per-block.

### FFN1Executor::on_complete
```cuda
atomicSub(&ffn1_done[slot], 1) == 1  → true if I'm the last FFN1 tile
```
If last: push FFN2 tiles into the task queue.

### FFN2Executor::execute
```cuda
gemv_tile_accumulate(down_proj, act, output, I, row_begin, row_count, weight)
```
`output[r] += weight * dot(down_proj[r,:], act)` — accumulates directly
into the final output buffer. After all K experts, output has the
weighted sum.

---

## tasks/gemv.cuh — GEMV Primitives

### gemv_tile<TPB>(A, x, y, N, row_begin, row_count)
Matrix-vector multiply for a row range. Each warp handles one row.
Uses float4 vectorized loads and warp shuffle reduction.
**Overwrites** y.

### gemv_tile_accumulate<TPB>(A, x, y, N, row_begin, row_count, scale)
Same as above but **accumulates**: `y[row] += scale * dot(A[row,:], x)`.
Used for FFN2 where multiple experts sum into the same output.

### Key design:
- 1 warp = 1 output row
- float4 loads (16 bytes at a time, 4 FMAs per load)
- warp_reduce_sum via __shfl_down_sync
- Lane 0 writes the final result

---

## tasks/softmax_topk.cuh — Router

### softmax_topk_warp<E, K>(logits, ids_out, weights_out)
Runs in ONE warp (32 threads). Two stages:
1. **Top-K**: K rounds of warp-wide argmax. Each round finds the max,
   records it, masks it out. O(K * E/32) register ops.
2. **Softmax**: Numerically stable exp + normalize over K values.
   Only lane 0 writes final ids and weights.

---

## Kernel Arguments

```cuda
__global__ void flash_moe_kernel(
    float *input,            // [H] token hidden state — read-only
    float *output,           // [H] final result — zero-init, accumulates K expert outputs
    float *ffn1_out,         // [K * I] scratch — stores FFN1 activations per slot
    FlashMoe<float> model,   // all weight pointers (passed by value, small struct)
    TaskQueue<CAPACITY> *tQ, // global task ring buffer
    Doorbell *doorbells,     // [NUM_WORKERS] per-worker mailboxes
    int *status_queue,       // [NUM_WORKERS] READY/BUSY per worker
    int *ffn1_done)          // [K] fan-in counters, init to FFN1_TILES_PER_EXPERT
```

### `ffn1_out` — FFN1 activation scratch
- Shape: `[TOP_K * MOE_INTERMEDIATE_SIZE]` = `[8 * 768]`
- Laid out as K slots: `ffn1_out[slot * I .. slot * I + I]`
- FFN1 tiles write their rows into the correct slot
- FFN2 reads from it: `act = ffn1_out + slot * I`
- Must be zero-initialized before launch

### `ffn1_done` — Fan-in counters
- Shape: `[TOP_K]` = `[8]`
- Each entry initialized to `FFN1_TILES_PER_EXPERT` (8)
- Every worker finishing an FFN1 tile does `atomicSub(&ffn1_done[slot], 1)`
- The worker that sees `old == 1` is the last one — it pushes FFN2 tasks
- After fan-in fires, the counter is 0 and never touched again

---

## Data Flow Summary

```
input[H]
    │
    ├─ gemv_tile(router) ──► logits[E]
    │                           │
    │                    softmax_topk ──► ids[K], weights[K]
    │
    ├─ For each expert k (via task queue):
    │     FFN1: gemv(gate) + gemv(up) + silu_mul ──► act[I]
    │                                                  │
    │     FFN2: gemv_accumulate(down, act, weight) ──► output[H] +=
    │
    └─ output[H]  (weighted sum of K experts)
```

Each step is a Task in the queue. The scheduler assigns tasks to
workers. Workers execute and push follow-up tasks on fan-in completion.
