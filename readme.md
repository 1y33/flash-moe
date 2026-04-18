Flash Moe - RTX 4070 kernel
Based on flash-dmoe s


SM = 46
one SM -> schedules the tasks . Other Sms are workers getting tasks



Convention : 
Tilling on N dimension 
SM -> task




## Future: Split-K task decomposition

Instead of 1 task = 1 expert GEMV, split each GEMV along the reduction dimension (D) into K chunks.
Each chunk computes a partial result, then a REDUCE task sums them.

This keeps all 45 worker SMs busy even with few experts (e.g. topK=2 with 8 experts = only 2 tasks vs 45 SMs idle).

Task flow becomes a DAG:
- GEMV_PARTIAL (gate, split 0..K-1) + GEMV_PARTIAL (up, split 0..K-1)
- REDUCE(gate) + REDUCE(up)
- SILU_MUL
- GEMV_PARTIAL (down, split 0..K-1)
- REDUCE(down)
- GATHER

Dependencies tracked via per-operation atomic counters. Last SM to finish a wave pushes the next task.


INFO: https://claude.ai/chat/d40a947c-eace-4ad7-8f19-a4d7ec6bbf79