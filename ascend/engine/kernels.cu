// ASCEND engine kernels: one simulation step = update (+ threshold, reset, stimulus), deliver, clear.
//
// The same source builds two ways:
//   * for NVIDIA GPUs by NVRTC through CuPy (ascend/engine/cuda.py);
//   * for the CPU by a C++ compiler with -DASCEND_EMULATE (ascend/engine/emulate.py), which runs
//     every thread of every kernel in turn so CI can test this exact logic without a GPU.
// Each kernel's work lives in a *_body(tid, nthreads, ...) function; the wrappers only supply tid.
//
// REAL is float or double, set at build time. Order of operations within a step matches
// ascend/engine/cpu.py and Brian2's schedule for the reference model.

#ifdef ASCEND_EMULATE
#define DEVICE static inline
template <typename T> static inline T atomicAdd(T* a, T v) { T old = *a; *a += v; return old; }
#else
#define DEVICE __device__ __forceinline__
#endif

typedef REAL real;
typedef unsigned long long u64;
typedef long long i64;

// rng = {seed, threshold}, kept in device memory so one captured graph serves every seed.
// rng.py: SplitMix64 finaliser on (seed + 1) * golden + (step << 20) + neuron.
DEVICE u64 mix(u64 x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

DEVICE bool kick(u64 seed, i64 step, int neuron, u64 thr) {
    u64 key = (seed + 1ULL) * 0x9E3779B97F4A7C15ULL + ((u64)step << 20) + (u64)neuron;
    return (mix(key) >> 32) < thr;
}

// Integrate, threshold, reset and stimulate every neuron for one step.
// Spikes are appended to ring slot (step + delay) % (delay + 1) and to the spike record.
DEVICE void update_body(i64 tid, i64 nthreads, int n, const i64* base_step, int k,
                        real* v, real* g, i64* last, const int* ref, unsigned char* nr_out,
                        const unsigned char* is_stim, const u64* rng,
                        real decay_g, real offset_v, real g_to_v, real decay_v,
                        real v_th, real v_rst, real w_poi,
                        int delay, int* ring_buf, int* ring_cnt,
                        i64* rec_step, int* rec_neuron, unsigned int* rec_cnt, unsigned int rec_cap) {
    const i64 step = *base_step + k;
    const u64 seed = rng[0], thr = rng[1];
    for (i64 i = tid; i < n; i += nthreads) {
        bool nr = (step - last[i]) >= ref[i];
        real gi = g[i], vi = v[i];
        if (nr) {
            real g_new = decay_g * gi;
            real v_new = offset_v + ((g_to_v * gi) + (decay_v * vi));
            gi = g_new;
            vi = v_new;
        }
        if (nr && vi > v_th) {
            last[i] = step;
            nr = false;
            const int slot = (int)((step + delay) % (delay + 1));
            const int pos = atomicAdd(&ring_cnt[slot], 1);
            ring_buf[(i64)slot * n + pos] = (int)i;
            const unsigned int r = atomicAdd(rec_cnt, 1u);
            if (r < rec_cap) {
                rec_step[r] = step;
                rec_neuron[r] = (int)i;
            }
            vi = v_rst;  // reset; nothing else touches a refractory neuron this step
            gi = 0;
        } else if (nr && is_stim[i] && kick(seed, step, (int)i, thr)) {
            vi += w_poi;
        }
        v[i] = vi;
        g[i] = gi;
        nr_out[i] = nr;
    }
}

// Deliver spikes emitted `delay` steps ago: one warp per spike, lanes stride over its synapses.
DEVICE void deliver_body(i64 tid, i64 nthreads, int n, const i64* base_step, int k, int delay,
                         const int* ring_buf, const int* ring_cnt, const i64* row_ptr, const int* post,
                         const real* w, const unsigned char* nr, real* g) {
    const i64 step = *base_step + k;
    const int slot = (int)(step % (delay + 1));
    const int cnt = ring_cnt[slot];
    const i64 lane = tid % 32, warp = tid / 32, nwarps = nthreads / 32;
    for (i64 j = warp; j < cnt; j += nwarps) {
        const int pre = ring_buf[(i64)slot * n + j];
        for (i64 s = row_ptr[pre] + lane; s < row_ptr[pre + 1]; s += 32) {
            const int q = post[s];
            if (nr[q]) atomicAdd(&g[q], w[s]);
        }
    }
}

DEVICE void clear_body(const i64* base_step, int k, int delay, int* ring_cnt) {
    ring_cnt[(int)((*base_step + k) % (delay + 1))] = 0;
}

#ifndef ASCEND_EMULATE
#define TID ((i64)blockIdx.x * blockDim.x + threadIdx.x)
#define NTHREADS ((i64)gridDim.x * blockDim.x)

extern "C" __global__ void update(int n, const i64* base_step, int k, real* v, real* g, i64* last,
                                  const int* ref, unsigned char* nr_out, const unsigned char* is_stim,
                                  const u64* rng, real decay_g, real offset_v, real g_to_v,
                                  real decay_v, real v_th, real v_rst, real w_poi, int delay,
                                  int* ring_buf, int* ring_cnt, i64* rec_step, int* rec_neuron,
                                  unsigned int* rec_cnt, unsigned int rec_cap) {
    update_body(TID, NTHREADS, n, base_step, k, v, g, last, ref, nr_out, is_stim, rng, decay_g,
                offset_v, g_to_v, decay_v, v_th, v_rst, w_poi, delay, ring_buf, ring_cnt, rec_step,
                rec_neuron, rec_cnt, rec_cap);
}

extern "C" __global__ void deliver(int n, const i64* base_step, int k, int delay, const int* ring_buf,
                                   const int* ring_cnt, const i64* row_ptr, const int* post,
                                   const real* w, const unsigned char* nr, real* g) {
    deliver_body(TID, NTHREADS, n, base_step, k, delay, ring_buf, ring_cnt, row_ptr, post, w, nr, g);
}

extern "C" __global__ void clear(const i64* base_step, int k, int delay, int* ring_cnt) {
    if (TID == 0) clear_body(base_step, k, delay, ring_cnt);
}

extern "C" __global__ void advance(i64* base_step, int steps) {
    if (TID == 0) *base_step += steps;
}
#endif
