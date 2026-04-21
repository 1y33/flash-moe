#!/bin/bash
# Sweep TILE_ROWS and THREADS_PER_BLOCK to find optimal config
# TILE_ROWS must divide MOE_INTERMEDIATE_SIZE (768) evenly
# THREADS_PER_BLOCK must be a multiple of 32

NVCC="nvcc"
ARCH="-arch=sm_89 -O3"
SRC="tests/test_kernel.cu"
BIN="build/test_kernel_sweep"

# TILE_ROWS candidates: must divide 768 evenly
TILES=(48 64 96 128 192 256 384 768)
# TPB candidates
TPBS=(32 64 128 256)

printf "%-8s %-6s %-8s %-8s %-10s\n" "TILE" "TPB" "FFN1_T" "FFN2_T" "TIME_MS"
printf "%-8s %-6s %-8s %-8s %-10s\n" "----" "---" "------" "------" "-------"

for TILE in "${TILES[@]}"; do
    for TPB in "${TPBS[@]}"; do
        # TILE must divide 768
        if (( 768 % TILE != 0 )); then
            continue
        fi
        # Need at least 1 warp
        if (( TPB < 32 )); then
            continue
        fi
        # ILP=4 needs TILE >= 4 * (TPB/32) for at least one iteration
        WARPS=$((TPB / 32))
        if (( TILE < WARPS * 4 )); then
            continue
        fi

        FFN1_T=$((768 / TILE))
        FFN2_T=$(( (2048 + TILE - 1) / TILE ))
        TOTAL=$(( 8 * (FFN1_T + FFN2_T) ))

        # Compile
        $NVCC $ARCH -DTILE_ROWS_OVERRIDE=$TILE -DTPB_OVERRIDE=$TPB -o $BIN $SRC 2>/dev/null
        if [ $? -ne 0 ]; then
            printf "%-8d %-6d %-8s %-8s %-10s\n" $TILE $TPB "-" "-" "COMPILE_ERR"
            continue
        fi

        # Run with timeout
        OUTPUT=$(timeout 10 ./$BIN 2>&1)
        if [ $? -ne 0 ]; then
            printf "%-8d %-6d %-8d %-8d %-10s\n" $TILE $TPB $FFN1_T $FFN2_T "TIMEOUT"
            continue
        fi

        # Extract kernel time
        TIME=$(echo "$OUTPUT" | grep "Kernel time:" | awk '{print $3}')
        PASS=$(echo "$OUTPUT" | grep -c "PASS")

        if [ "$PASS" -gt 0 ] && [ -n "$TIME" ]; then
            printf "%-8d %-6d %-8d %-8d %-10s\n" $TILE $TPB $FFN1_T $FFN2_T "$TIME"
        else
            printf "%-8d %-6d %-8d %-8d %-10s\n" $TILE $TPB $FFN1_T $FFN2_T "FAIL"
        fi
    done
done
