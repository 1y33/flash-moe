// Trace library generated using CC :D
//
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <climits>

// ─── Data types ──────────────────────────────────────────────

struct TraceEvent {
    int label;
    int block_id;
    int sm_id;
    int warp_id;
    long long start;
    long long end;
};

constexpr int TRACE_MAX_WARPS = 8;

struct DeviceTracer {
    TraceEvent* buf;
    int* count;
    int max_events;

#ifdef TRACE_ENABLED
    static __device__ __forceinline__ int get_sm_id() {
        int sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        return sm;
    }

    static __device__ __forceinline__ long long globaltimer() {
        long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        return t;
    }

    __device__ __forceinline__ void start(int label, long long* pending) {
        int wid = threadIdx.x / 32;
        pending[wid] = globaltimer();
    }

    __device__ __forceinline__ void stop(int label, long long* pending) {
        long long t = globaltimer();
        int wid = threadIdx.x / 32;
        int idx = atomicAdd(count, 1);
        if (idx < max_events) {
            buf[idx].label    = label;
            buf[idx].block_id = blockIdx.x;
            buf[idx].sm_id    = get_sm_id();
            buf[idx].warp_id  = wid;
            buf[idx].start    = pending[wid];
            buf[idx].end      = t;
        }
    }
#else
    __device__ __forceinline__ void start(int, long long*) {}
    __device__ __forceinline__ void stop(int, long long*) {}
#endif
};

// ─── Host utilities ──────────────────────────────────────────

struct TraceBuffer {
    static void allocate(TraceEvent** buf, int** count, int max_events) {
#ifdef TRACE_ENABLED
        cudaMalloc(buf, max_events * sizeof(TraceEvent));
        cudaMalloc(count, sizeof(int));
        cudaMemset(*count, 0, sizeof(int));
#else
        *buf = nullptr;
        *count = nullptr;
#endif
    }

    static void free(TraceEvent* buf, int* count) {
#ifdef TRACE_ENABLED
        cudaFree(buf);
        cudaFree(count);
#endif
    }

    static void print(TraceEvent* d_buf, int* d_count,
                      const char** label_names, int num_labels) {
#ifdef TRACE_ENABLED
        int h_count;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_count == 0) {
            printf("[trace] No events recorded.\n");
            return;
        }

        TraceEvent* events = (TraceEvent*)malloc(h_count * sizeof(TraceEvent));
        cudaMemcpy(events, d_buf, h_count * sizeof(TraceEvent), cudaMemcpyDeviceToHost);

        struct LabelStats {
            long long total;
            long long min_cy;
            long long max_cy;
            int count;
        };

        LabelStats* stats = (LabelStats*)calloc(num_labels, sizeof(LabelStats));
        for (int i = 0; i < num_labels; i++) {
            stats[i].min_cy = LLONG_MAX;
            stats[i].max_cy = 0;
        }

        long long global_total = 0;

        for (int i = 0; i < h_count; i++) {
            int l = events[i].label;
            if (l < 0 || l >= num_labels) continue;
            long long dur = events[i].end - events[i].start;
            stats[l].total  += dur;
            stats[l].count  += 1;
            if (dur < stats[l].min_cy) stats[l].min_cy = dur;
            if (dur > stats[l].max_cy) stats[l].max_cy = dur;
            global_total += dur;
        }

        printf("\n╔══════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  GPU TRACE SUMMARY  (%d events)                                         ║\n", h_count);
        printf("╠═══════════════╦════════╦═══════════════╦═══════════════╦═══════════════╦══╣\n");
        printf("║ %-13s ║  cnt   ║   avg cycles  ║   min cycles  ║   max cycles  ║ %%║\n", "label");
        printf("╠═══════════════╬════════╬═══════════════╬═══════════════╬═══════════════╬══╣\n");

        for (int l = 0; l < num_labels; l++) {
            if (stats[l].count == 0) continue;
            long long avg = stats[l].total / stats[l].count;
            double pct = global_total > 0 ? 100.0 * stats[l].total / global_total : 0.0;
            printf("║ %-13s ║ %6d ║ %13lld ║ %13lld ║ %13lld ║%2.0f║\n",
                   label_names[l], stats[l].count, avg,
                   stats[l].min_cy, stats[l].max_cy, pct);
        }

        printf("╚═══════════════╩════════╩═══════════════╩═══════════════╩═══════════════╩══╝\n\n");

        ::free(stats);
        ::free(events);
#endif
    }

    static void write_json(TraceEvent* d_buf, int* d_count,
                           const char** label_names, int num_labels,
                           const char* filename,
                           int first_leaf = -1) {
#ifdef TRACE_ENABLED
        int h_count;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_count == 0) {
            printf("[trace] No events to write.\n");
            return;
        }

        TraceEvent* events = (TraceEvent*)malloc(h_count * sizeof(TraceEvent));
        cudaMemcpy(events, d_buf, h_count * sizeof(TraceEvent), cudaMemcpyDeviceToHost);

        long long t_min = LLONG_MAX;
        for (int i = 0; i < h_count; i++) {
            if (events[i].start < t_min) t_min = events[i].start;
        }

        FILE* f = fopen(filename, "w");
        if (!f) {
            printf("[trace] Failed to open %s\n", filename);
            ::free(events);
            return;
        }

        fprintf(f, "{\n  \"first_leaf\": %d,\n  \"labels\": [", first_leaf);
        for (int l = 0; l < num_labels; l++) {
            fprintf(f, "\"%s\"%s", label_names[l], l < num_labels - 1 ? ", " : "");
        }
        fprintf(f, "],\n  \"events\": [\n");

        for (int i = 0; i < h_count; i++) {
            int l = events[i].label;
            const char* name = (l >= 0 && l < num_labels) ? label_names[l] : "?";
            fprintf(f, "    {\"l\":%d,\"n\":\"%s\",\"b\":%d,\"sm\":%d,\"w\":%d,\"s\":%lld,\"e\":%lld}%s\n",
                    events[i].label, name,
                    events[i].block_id, events[i].sm_id, events[i].warp_id,
                    events[i].start - t_min, events[i].end - t_min,
                    i < h_count - 1 ? "," : "");
        }

        fprintf(f, "  ]\n}\n");
        fclose(f);
        printf("[trace] Wrote %d events to %s\n", h_count, filename);

        ::free(events);
#endif
    }
};
