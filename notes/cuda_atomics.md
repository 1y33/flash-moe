# CUDA Atomics Reference

## What are atomics?

Atomics are read-modify-write operations that are guaranteed to complete
without interference from other threads. Without atomics, two threads
doing `x += 1` can both read `x=0`, both compute `1`, both write `1`,
and you lose an increment.

## The atomics we use

### `atomicAdd(addr, val)` → returns old value
```cuda
int old = atomicAdd(&counter, 1);
// counter is now old+1, exactly one thread got each old value
```
Used for: queue head advancement, counting completed tasks.

### `atomicSub(addr, val)` → returns old value
```cuda
int before = atomicSub(&counter, 1);
if (before == 1) {
    // I decremented from 1 to 0 — I'm the LAST one
    // Push the next stage of work
}
```
Used for: fan-in synchronization. Initialize counter to N, each parent
decrements. Exactly one thread sees `before == 1`.

### `atomicExch(addr, val)` → returns old value
```cuda
int old = atomicExch(&status, BUSY);
if (old == READY) {
    // I successfully claimed this slot
}
```
Used for: claiming workers (scheduler), consuming doorbells (workers).

Difference from `atomicAdd`: exchange REPLACES the value entirely,
it doesn't add to it. Think of it as "swap".

### `atomicCAS(addr, compare, val)` → returns old value
```cuda
int old = atomicCAS(&tail, expected, expected + 1);
if (old == expected) {
    // I successfully advanced tail from expected to expected+1
}
```
"Compare And Swap" — only writes if the current value matches `compare`.
Used in lock-free queues when multiple consumers compete.

We don't use CAS much because our scheduler is single-consumer (no
contention on pop).

## Memory ordering: `__threadfence()`

Atomics guarantee the operation itself is atomic, but NOT that other
memory writes before it are visible. `__threadfence()` ensures all
prior writes by this thread are visible to all other threads on the
device before any subsequent writes.

### The doorbell pattern:
```
Thread A (scheduler):           Thread B (worker):
  doorbells[w].task_idx = 42;     // might see stale task_idx!
  __threadfence();                // ensures task_idx is flushed
  atomicExch(&doorbells[w].ready, 1);
                                  // polls ready, sees 1
                                  // reads task_idx → guaranteed 42
```

Without `__threadfence()`, thread B might see `ready=1` but read
garbage from `task_idx` because the write hasn't propagated yet.

### When you need it:
- Before signaling (writing a flag/counter that others poll)
- After writing data that another SM will read
- Before `atomicSub` in fan-in (ensures computation output is visible)

### When you DON'T need it:
- Between operations on the same thread (already ordered)
- Between threads in the same warp (warp is lockstep)
- For `__shared__` memory (use `__syncthreads()` instead)

## `__syncthreads()` vs `__syncwarp()` vs `__threadfence()`

| Primitive | Scope | What it does |
|-----------|-------|-------------|
| `__syncthreads()` | Block | Barrier: ALL threads in block must reach it. Also acts as memory fence for shared memory. |
| `__syncwarp()` | Warp (32 threads) | Barrier for threads within one warp. Lighter weight. |
| `__threadfence()` | Device | Memory fence only (no barrier). Ensures writes are visible to other blocks. |

**Critical rule**: `__syncthreads()` must be reached by ALL threads in
the block. Never put it inside an `if` that excludes some threads.

## Atomic performance

Atomics to the same address serialize — they create contention.

| Pattern | Contention | Notes |
|---------|-----------|-------|
| 45 threads atomicAdd to same counter | High | All 45 serialize |
| 45 threads atomicExch on 45 different slots | None | Each touches its own slot |
| 1 thread pops from queue, 45 push | Low | 1 reader, few concurrent writers |

Our design minimizes contention:
- Queue pop: only the scheduler (1 thread) → zero contention
- Queue push: bootstrap (1 thread) + workers (sporadic, on fan-in) → low
- Status queue: each worker writes its own slot → zero contention
- Doorbell: scheduler writes, worker reads different slots → zero contention
