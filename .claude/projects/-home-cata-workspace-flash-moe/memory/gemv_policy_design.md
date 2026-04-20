# GEMV Policy-Based Design

## Goal
Swap GEMV implementations (ILP, pipelined, etc.) by changing one config line.

## File Structure
```
csrc/
  flashmoe.cuh              ← GemvImpl enum + GEMV_POLICY constant
  tasks/
    gemv.cuh                 ← thin include: pulls selector, exposes `using Gemv = ...`
    gemv/
      interface.cuh          ← documents required methods (reference only)
      ilp.cuh                ← current ILP implementation
      base.cuh               ← simple ILP=1 fallback
      pipelined.cuh          ← future cp.async version
      selector.cuh           ← maps enum → struct via template specialization
```

## Interface Contract
Every `GemvXxx` struct must provide these static device methods:

```cuda
struct GemvXxx {
    template <typename T, int TPB = 128, int ILP = 4>
    static __device__ void tile(
        const T* A, const T* x, float* y,
        int N, int row_begin, int row_count);

    template <typename T, int TPB = 128, int ILP = 4>
    static __device__ void tile_accumulate(
        const T* A, const T* x, float* y,
        int N, int row_begin, int row_count, float scale);

    template <typename T, int TPB = 128, int ILP = 4>
    static __device__ void tile_accumulate_mixed(
        const T* A, const float* x, float* y,
        int N, int row_begin, int row_count, float scale);

    template <typename T, int TPB = 128, int ILP = 4>
    static __device__ void fused_gate_up(
        const T* A_gate, const T* A_up, const T* x,
        float* y_gate, float* y_up,
        int N, int row_begin, int row_count);
};
```

## Selector (gemv/selector.cuh)
```cuda
enum class GemvImpl { BASE, ILP, PIPELINED };

template <GemvImpl impl> struct GemvSelect;
template <> struct GemvSelect<GemvImpl::BASE>      : GemvBase {};
template <> struct GemvSelect<GemvImpl::ILP>       : GemvILP {};
// template <> struct GemvSelect<GemvImpl::PIPELINED> : GemvPipelined {};
```

## Config (flashmoe.cuh)
```cuda
constexpr GemvImpl GEMV_POLICY = GemvImpl::ILP;  // ← change this to switch
```

## Top-level gemv.cuh
```cuda
#pragma once
#include "gemv/selector.cuh"
using Gemv = GemvSelect<GEMV_POLICY>;
```

## Usage in executor
```cuda
Gemv::fused_gate_up<T, 128>(gate_proj, up_proj, input, act, up_smem, N, row_begin, row_count);
Gemv::tile_accumulate_mixed<T, 128>(down_proj, act, output, N, row_begin, row_count, weight);
```

## Adding a new variant
1. Create `gemv/my_impl.cuh` with `struct GemvMyImpl { ... same methods ... }`
2. `#include "my_impl.cuh"` in selector.cuh
3. Add `template <> struct GemvSelect<GemvImpl::MY_IMPL> : GemvMyImpl {};`
4. Set `GEMV_POLICY = GemvImpl::MY_IMPL` in flashmoe.cuh
5. Rebuild

## Extends to other components
Same pattern works for Scheduler, SiluMul, Executor — can unify into a single KernelConfig struct:
```cuda
struct KernelConfig {
    static constexpr GemvImpl      gemv      = GemvImpl::ILP;
    static constexpr SchedulerImpl scheduler = SchedulerImpl::SINGLE;
    static constexpr int           TILE_ROWS = 96;
    static constexpr int           ILP       = 4;
    static constexpr int           TPB       = 128;
};
```

## Key properties
- Zero runtime cost — all resolved at compile time via struct inheritance + templates
- If a variant is missing a method, you get a compile error (type-safe)
- Can build multiple binaries with different policies via -D flags