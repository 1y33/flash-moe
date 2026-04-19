# Bugs We Hit and How We Fixed Them

## Bug 1: `__syncthreads()` inside warp-conditional branch → Deadlock

**Symptom**: Kernel hangs forever (timeout).

**Code**:
```cuda
if (warp_id == 0) {
    BootStrap::route(...);
    __syncthreads();   // ← DEADLOCK
    BootStrap::dispatch(...);
}
else if (warp_id == 1) {
    Scheduler::run(...);
}
```

**Why**: `__syncthreads()` requires ALL threads in the block to reach it.
Warp 0 enters the `if` and hits `__syncthreads()`. Warp 1 enters the
`else` and never reaches it. Both sides wait forever.

**Fix**: Move the `__syncthreads()` OUTSIDE the branch so all threads
participate:
```cuda
// ALL threads do GEMV (needs full block anyway)
BootStrap::route(...);
__syncthreads();  // everyone hits this

if (warp_id == 0) {
    // topk + dispatch (warp-only ops)
}
else if (warp_id == 1) {
    // scheduler
}
```

**Rule**: Never put `__syncthreads()` inside an `if` that excludes
some threads. Use `__syncwarp()` for warp-internal sync instead.

---

## Bug 2: `__shared__` variables as struct members → Ignored

**Symptom**: Compiler warning: `'shared' attribute directive ignored`.

**Code**:
```cuda
struct OS {
    __shared__ float logits[128];  // ← ignored, becomes regular member
    ...
};
```

**Why**: CUDA doesn't support `__shared__` on struct members. The
`__shared__` qualifier is silently dropped, making it a regular (stack)
variable. On the GPU stack space is tiny, so this either corrupts
memory or crashes.

**Fix**: Declare `__shared__` variables as local variables inside a
`__device__` function:
```cuda
struct OS {
    static __device__ void run(...) {
        __shared__ float logits[128];  // ← correct
        ...
    }
};
```

---

## Bug 3: Shared scratch buffer collision across workers → Wrong results

**Symptom**: Output values wrong (max abs diff ~0.04 vs reference).
Routing was correct (same experts, same weights). Bug was in FFN computation.

**Code**:
```cuda
// All workers share the same scratch space for "up" GEMV results
float *up_scratch = ffn1_out + TOP_K * I + task.row_begin;
```

**Why**: Multiple worker blocks run FFN1 tiles concurrently. If two
workers process tiles with the same `row_begin` (even from different
experts), they write to the same location in `up_scratch`. The values
get stomped.

**Fix**: Use block-private `__shared__` memory instead:
```cuda
__shared__ float up_smem[TILE_ROWS];  // private to this block
// Each worker has its own copy, no collision
```

**Rule**: If multiple blocks write to the same global buffer concurrently,
you need either per-block regions or atomics. `__shared__` memory is
always block-private, making it safe.

---

## Bug 4: Task queue CAPACITY overflow → Undefined behavior

**Symptom**: Not directly observed (caught during review).

**Why**: CAPACITY was 128. FFN1 tasks = 8 experts × 8 tiles = 64.
FFN2 tasks = 8 experts × 22 tiles = 176. Total = 240 > 128.
The ring buffer wraps around and overwrites active tasks.

**Fix**: Increased CAPACITY to 512 (next power of 2 above 240).
The ring buffer uses `idx & (capacity - 1)`, which requires capacity
to be a power of 2.

---

## Bug 5: `FFN2_TILES_PER_EXPERT` integer division truncation

**Symptom**: Not directly observed (caught during review).

**Code**:
```cuda
constexpr int FFN2_TILES_PER_EXPERT = HIDDEN_SIZE / TILE_ROWS;  // 2048/96 = 21
```

**Why**: 2048 / 96 = 21.33. Integer division gives 21, but `push_ffn2_tasks`
uses `min(TILE_ROWS, rows_left)` which correctly creates a 22nd tile
with 32 rows. So the actual count is 22 but `TOTAL_TASKS` used 21,
causing the scheduler to exit before all tasks were processed.

**Fix**: Use ceiling division:
```cuda
constexpr int FFN2_TILES_PER_EXPERT = (HIDDEN_SIZE + TILE_ROWS - 1) / TILE_ROWS;  // 22
```

---

## Bug 6: Separate kernel launches for doorbell test → Sequential execution

**Symptom**: Test hangs (reader kernel spins forever).

**Code**:
```cuda
test_doorbell_reader<<<1, 1>>>(db, results);  // launch 1
test_doorbell_writer<<<1, 1>>>(db);            // launch 2
```

**Why**: CUDA kernel launches on the same stream are sequential.
The reader launches first, spins waiting for `ready == 1`, but the
writer hasn't launched yet because it's queued behind the reader.

**Fix**: Combine into one kernel with 2 blocks:
```cuda
__global__ void test_doorbell(Doorbell *db, int *results) {
    if (blockIdx.x == 0) { /* writer */ }
    else                 { /* reader */ }
}
test_doorbell<<<2, 1>>>(db, results);  // both blocks run concurrently
```
