# Doorbell System: How Tasks Get to Workers

## The Problem

We have 1 OS block and 45 worker blocks. The OS needs to tell workers
"here's your next task". Workers need to tell the OS "I'm free, give me work".

We can't use shared memory (it's per-block). So everything goes through
**global memory** with atomics.

---

## Three Global Arrays

```
status_queue[45]     doorbells[45]         task_queue[128]
┌───┐                ┌──────────┐          ┌──────┐
│ 0 │ READY          │ idx: 7   │          │ T0   │
│ 1 │ BUSY           │ ready: 0 │          │ T1   │
│ 2 │ READY          │ idx: -   │          │ T2   │
│ 3 │ BUSY           │ ready: 1 │ ←bell!   │ T3   │
│...│                │ ...      │          │ ...  │
└───┘                └──────────┘          └──────┘
```

### 1. `status_queue[P]` — "who is free?"

Each worker has one slot. Two values:
- `PROC_READY (0)` — worker is idle, available for work
- `PROC_BUSY (1)` — worker is executing a task

**Writer**: the worker (sets READY when done with a task)
**Reader**: the scheduler (atomicExch to BUSY when claiming a worker)

### 2. `doorbells[P]` — "here's your task"

Each worker has a doorbell. It contains:
- `task_idx` — index into `task_queue` where the task lives
- `ready` — signal flag:
  - `0` = no task (worker spins on this)
  - `1` = task available (worker wakes up)
  - `2` = EXIT (worker shuts down)

**Writer**: the scheduler (writes task_idx, then sets ready=1)
**Reader**: the worker (polls ready, reads task_idx when ready=1)

### 3. `task_queue[capacity]` — "all pending tasks"

A ring buffer of `Task` structs. Multiple producers, single consumer:
- **Producers**: bootstrap (pushes FFN1 tasks), workers (push FFN2 tasks after fan-in)
- **Consumer**: scheduler (pops tasks to assign to workers)

---

## The Flow, Step by Step

### Phase 1: Bootstrap fills the queue

```
Bootstrap (OS block, warp 0):

  1. router GEMV → logits[128]
  2. softmax_topk → 8 expert IDs + weights
  3. For each expert, split FFN1 into row tiles:
     push Task{expert=42, type=FFN1, row_begin=0,  row_count=96} → task_queue
     push Task{expert=42, type=FFN1, row_begin=96, row_count=96} → task_queue
     ... (8 tiles per expert × 8 experts = 64 tasks)
```

### Phase 2: Scheduler assigns tasks

```
Scheduler (OS block, warp 1, thread 0 only):

  loop:
    1. Pop a task from task_queue
    2. Scan status_queue for a READY worker:
       old = atomicExch(&status_queue[p], PROC_BUSY)
       if old == PROC_READY → found one!
    3. Write task index into that worker's doorbell:
       doorbells[p].task_idx = <index of the task in task_queue>
       __threadfence()                    ← make write visible
       atomicExch(&doorbells[p].ready, 1) ← ring the bell
    4. Repeat until all tasks scheduled
```

### Phase 3: Worker executes

```
Worker p (block p, all threads):

  loop:
    1. Thread 0 polls doorbell:
       while (doorbells[p].ready == 0) {} ← spin
       signal = atomicExch(&doorbells[p].ready, 0) ← consume

    2. If signal == 2: EXIT, break

    3. Read task from task_queue using doorbells[p].task_idx

    4. Broadcast task to all threads via shared memory

    5. All threads cooperatively execute:
       - FFN1: partial gate GEMV + up GEMV + silu_mul
       - FFN2: partial down GEMV with accumulate

    6. Thread 0: mark_ready → atomicExch(&status_queue[p], PROC_READY)

    7. Back to step 1
```

---

## Fan-in: How FFN2 Tasks Get Created

FFN1 for one expert is split into 8 tiles (8 workers can work on it
in parallel). But FFN2 can't start until ALL 8 FFN1 tiles for that
expert are done.

Solution: `ffn1_done[TOP_K]` — one atomic counter per expert slot,
initialized to `FFN1_TILES_PER_EXPERT` (8).

```
Worker finishes an FFN1 tile:
  before = atomicSub(&ffn1_done[slot], 1)
  if before == 1:
      → I'm the LAST tile for this expert
      → Push FFN2 tiles into task_queue
      → Scheduler will pick them up and assign to workers
```

Only ONE worker ever sees `before == 1` (atomicSub guarantees this).
That worker becomes the one who creates the next stage of work.

```
                FFN1 tiles for expert k
        ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
        │ W3  │ W7  │ W12 │ W1  │ W22 │ W9  │ W15 │ W40 │
        └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
           │     │     │     │     │     │     │     │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                          all decrement
                        ffn1_done[k] -= 1
                              │
                    last one (before==1)
                              │
                    pushes FFN2 tiles ──→ task_queue
```

---

## Why Not Just Let Workers Pop From the Queue?

That's the "worker-driven" model (what AMD's flash-moe does).
We use "scheduler-driven" because:

1. **No contention**: 45 workers all doing atomicCAS on the same queue
   head = contention. One scheduler thread popping = zero contention.

2. **Visibility**: The scheduler sees the global picture. It knows who's
   free and what work exists. Workers only know about themselves.

3. **Policy in one place**: Want to prioritize FFN2 over FFN1? Change
   the scheduler. Workers don't change. Clean separation.

4. **Cost**: We sacrifice 1 warp (32 threads) to be the scheduler.
   That's ~2% of a 4070. Worth it for the simplicity.

---

## Memory Ordering

The doorbell protocol has a subtle ordering requirement:

```
Scheduler:                          Worker:
  doorbells[p].task_idx = 42        while (doorbells[p].ready == 0) {}
  __threadfence()          ←─── ensures task_idx is visible
  doorbells[p].ready = 1   ───→ worker reads ready=1
                                 reads task_idx → guaranteed to see 42
```

Without `__threadfence()`, the worker might see `ready=1` but read
a stale `task_idx`. The fence ensures the task data is globally visible
before the bell rings.

Same pattern when workers push FFN2 tasks:

```
Worker:
  write FFN1 output to global memory
  __threadfence()          ← ensures FFN1 output is visible
  atomicSub(&ffn1_done[k], 1)  ← only then signal completion
```

---

## Summary

```
┌─────────────────────────────────────────────────────────┐
│                    OS Block (block 0)                     │
│                                                           │
│  Warp 0: Bootstrap                                        │
│    route → softmax_topk → push FFN1 tasks into task_queue │
│                                                           │
│  Warp 1: Scheduler                                        │
│    loop: pop task_queue → find READY worker → ring bell    │
└─────────────────────────────────────────────────────────┘
              │ doorbell[p].ready = 1
              ▼
┌─────────────────────────────────────────────────────────┐
│                  Worker Block (block p)                    │
│                                                           │
│  poll doorbell → load task → execute → mark READY         │
│  if last FFN1 tile: push FFN2 tasks back to task_queue    │
└─────────────────────────────────────────────────────────┘
```
