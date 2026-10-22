# 07 — GPU Architecture & the CUDA Programming Model

> **Mental model:** *a GPU is a throughput machine that hides memory latency with massive parallelism. Threads run in groups of 32 (**warps**) that execute one instruction at a time in lockstep. While one warp waits for memory, the hardware switches to another for free. Your job is to give it **enough warps** (occupancy), make each warp touch **few memory sectors** (coalescing), and keep each warp's threads on the **same path** (low divergence).*

---

## 1. CPU vs GPU: two answers to the memory wall

```
 CPU: latency-optimised                         GPU: throughput-optimised
 ┌──────────────────────────────┐               ┌──────────────────────────────────────────┐
 │ ┌────────┐ ┌────────┐        │               │ SM  SM  SM  SM  SM  SM  SM  SM            │
 │ │ big    │ │ big    │  few   │               │ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓  many     │
 │ │ core + │ │ core + │  fat   │               │ SM  SM  SM  SM  SM  SM  SM  SM   simple   │
 │ │ OoO,   │ │ branch │  cores │               │ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓  lanes    │
 │ │ predict│ │ predict│        │               │ small caches per SM; huge register files │
 │ └────────┘ └────────┘        │               └──────────────────────────────────────────┘
 │ big caches hide latency      │                switching warps hides latency
 └──────────────────────────────┘
 one thread: fast                               one thread: slow; 10,000s of threads: fast
```

A CPU makes *one* thread fast (out-of-order execution, branch prediction, big caches). A GPU makes *many* threads fast in aggregate by **always having another warp ready to run** while others wait on memory. That's why GPU code needs tens of thousands of threads to perform well.

---

## 2. The execution hierarchy: software ↔ hardware

```
 SOFTWARE (what you launch)                    HARDWARE (GTX 1650 Max-Q, Turing sm_75)
 grid ─── many blocks ───────────────────────▶ distributed across 16 SMs
   block (≤ 1024 threads, shares smem) ──────▶ runs entirely on ONE SM; ≤ 16 blocks resident per SM
     warp = 32 consecutive threads ──────────▶ scheduled as a unit by one of the SM's 4 warp schedulers
       thread ───────────────────────────────▶ one lane; has private registers

 render kernel: grid = 80 × 45 tiles (720p) = 3600 blocks, block = 16×16 = 256 threads = 8 warps
```

**Key facts:**

- **Blocks are independent.** No ordering between blocks, and no barrier across blocks inside a kernel. A kernel boundary is the only global barrier.
- **Threads in a block cooperate** via shared memory and `__syncthreads()`.
- **Warps are the real unit of execution.** Thread indices map to warps by linear thread id. For a 16×16 block, warp 0 covers rows 0–1 of the tile.
- **Per-SM limits on Turing:** 1024 resident threads (32 warps), 16 resident blocks, 64K 32-bit registers, up to 64 KB shared memory per block (out of a 96 KB L1/shared pool).

---

## 3. The memory hierarchy

```
               per thread      per block            per SM                  whole GPU
            ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   ┌───────────────────────┐
 fastest ─▶ │ registers  │  │ shared memory│  │ L1 (shares pool  │   │ L2 cache (~1 MB)      │
            │ ~0 latency │  │ ~20–30 cycles│  │  with shared mem)│   │ ~200 cycles           │
            │ 64K / SM   │  │ explicit,    │  │ automatic        │   ├───────────────────────┤
            └────────────┘  │ banked       │  └──────────────────┘   │ global memory (VRAM)  │
                            └──────────────┘                         │ 4 GB, ~400–600 cycles │
            constant cache: kernel parameters & __constant__ (broadcast when all lanes read one address)
            read-only/texture path: __ldg / const __restrict__ loads
```

(Latencies are order-of-magnitude figures for this generation; module 08 measures the real ones.)

| Space | Scope | Managed by | SplatScope use |
|---|---|---|---|
| Registers | Thread | Compiler | Every local variable; too many cause **spills** to local memory (which lives in DRAM, so it's slow) |
| Shared memory | Block | **You** | Render kernel's Gaussian batch (`s_xy`, `s_conic_o`, `s_rgb`) |
| L1 / L2 | SM / GPU | Hardware | Implicit reuse (SH reads, gathers) |
| Global | Grid + host | You (`cudaMalloc`) | All `DeviceBuffer`s |
| Constant / param | Grid | Compiler | `CameraParams`, `RenderConsts` passed by value |
| Local | Thread | Compiler (spills, dynamically indexed arrays) | Avoid: check `-Xptxas -v` for "spill stores" |

---

## 4. Coalescing: how a warp's loads become memory transactions

A warp issues one load instruction for 32 lanes. The memory system serves it in **32-byte sectors**. What matters is **how many distinct sectors** the 32 addresses touch.

```
 (a) consecutive 4-byte floats: a[tid]
     lanes  0  1  2  3 … 31
     bytes |0..127|                            → 4 sectors for 128 useful bytes   ✅ 100 %
 (b) consecutive 16-byte Vec4f: a[tid] (SplatScope mean_opacity)
     bytes |0 ………………… 511|                     → 16 sectors for 512 useful bytes  ✅ 100 %
 (c) stride 2: a[2*tid]
     bytes |0.._.._.._ … 255|                   → 8 sectors for 128 useful bytes   ⚠ 50 %
 (d) stride ≥ 8 floats, or random gather: a[idx[tid]]
     bytes |s0|  |s1|     |s2| …               → up to 32 sectors for 128 useful bytes ❌ 12.5 %
 (e) all lanes read the SAME address
     → 1 sector, broadcast                                                       ✅
```

**SplatScope mapping:**

| Kernel access | Pattern | Quality |
|---|---|---|
| Preprocess reads `mean_opacity[i]`, `cov3d_a[i]` | (b) | ✅ |
| Preprocess reads 48 SH floats of Gaussian *i* | Each lane walks its own 192 B block (stride 192 B across lanes) | ⚠ First touch is scattered; later loads hit L1. Measure it |
| Duplicate writes `keys[off..off+k]` | Lanes write *different-length* runs at scattered offsets | ⚠ Inherent; fine |
| Render gathers `mean2d[point_list[k]]` | (d) random gather | ❌ Inherent, so do it **once per batch into shared memory**, then reuse it 256× |
| Render reads `s_xy[j]` (all lanes, same `j`) | (e) broadcast from shared memory | ✅ |

---

## 5. SIMT and divergence

All active lanes of a warp execute the **same instruction**. On a branch where lanes disagree, the warp runs **both** paths, with lanes masked off on the path they didn't take.

```
 if (power > 0) continue;  // lanes 0–9 skip, 10–31 compute
 ┌───────── warp timeline ─────────┐
 │ path A (lanes 0–9)   ▓▓          │   lanes 10–31 idle
 │ path B (lanes 10–31)   ▓▓▓▓▓▓▓▓  │   lanes 0–9 idle
 └──────────────────────────────────┘   cost ≈ A + B, not max(A, B)
```

**SplatScope divergence sources:**

1. **Early termination in render:** pixels finish at different times; finished lanes idle until the block votes `done`. That's inherent, and the reason the per-block vote (`__syncthreads_count`) exists.
2. **Different loop lengths in duplicate:** Gaussians touch 1 to hundreds of tiles; a warp runs as long as its biggest Gaussian. Measurable; fixable by load balancing (module 08).
3. **Culling in preprocess:** culled lanes exit early. Cheap, because the remaining work is short.

**Uniform branches cost nothing:** a branch on `settings.debug_view` is the same for every lane, so there's no divergence.

---

## 6. Latency hiding and occupancy

**Little's law** for memory: to sustain bandwidth B with latency T, you need **B × T bytes in flight**.

$$
\text{bytes in flight} \approx 192\ \tfrac{\text{GB}}{\text{s}} \times 500\ \text{ns} \approx 96\ \text{KB across the GPU} \;\Rightarrow\; \approx 6\ \text{KB per SM}
$$

Each warp has only a few loads in flight. So you need **many resident warps per SM**, or several independent loads per thread (instruction-level parallelism), to keep the memory system busy.

**Occupancy** = resident warps ÷ maximum warps per SM (32 on Turing). It's limited by whichever runs out first:

| Limiter (Turing per SM) | Render kernel (256 thr/block) at 48 regs | …at 80 regs | …at 40 KB smem/block |
|---|---|---|---|
| Threads (1024) | 4 blocks | 4 blocks | 4 blocks |
| Registers (65,536) | 65536 / (256·48) = 5.3 → 5 | 65536 / (256·80) = 3.2 → **3** | 5 |
| Shared memory (64 KB usable per SM for shared, depending on carveout) | many | many | 64 / 40 → **1** |
| Blocks (16) | 16 | 16 | 16 |
| **Resident blocks → occupancy** | **4 → 100 %** | **3 → 75 %** | **1 → 25 %** |

**Nuances an interviewer will probe:**

- **Higher occupancy isn't always faster.** If you're compute-bound with high ILP, 50% can match 100%. Occupancy is a means, not the goal.
- **Registers are the usual limiter.** Check with `-Xptxas -v` ("Used N registers"). Use `__launch_bounds__(256, 4)` to tell the compiler your target, which can force spills, so measure.
- **Spills** move variables to local memory (DRAM), which is often worse than lower occupancy.

---

## 7. Shared memory banks

Shared memory has **32 banks**, 4 bytes wide. Address `a` lives in bank `(a / 4) % 32`. In one access, lanes hitting **different addresses in the same bank** are serialised (a *bank conflict*). Lanes reading the **same address** get a broadcast.

```
 s[tid]        → lane i hits bank i         ✅ no conflict
 s[2*tid]      → lanes 0 and 16 share bank 0 ⚠ 2-way conflict
 s[32*tid]     → all lanes hit bank 0         ❌ 32-way conflict
 s[j] (same j) → broadcast                    ✅ (render kernel's inner loop)
```

The render kernel's pattern (cooperative load by `rank`, then all lanes read the same `j`) is conflict-free by construction. That's one reason it's structured that way.

---

## 8. Synchronisation and communication

| Scope | Primitive | SplatScope use |
|---|---|---|
| Warp | `__syncwarp`, `__shfl_sync`, `__ballot_sync`, `__any_sync` | Optional: warp-level early-exit votes |
| Block | `__syncthreads`, `__syncthreads_count/and/or` | Render batching + block-wide done vote |
| Grid | **Kernel boundary** (or cooperative groups grid sync, rarely used) | Between pipeline stages |
| Device ↔ host | Stream order, events, `cudaStreamSynchronize` | The L readback; Phase A image readback |
| Any | Atomics (`atomicAdd`, …) | Avoided by design: the scan gives deterministic offsets instead |

**Rule:** a `__syncthreads()` must be reached by **all** threads of the block, or by none. A conditional `return` before it is undefined behaviour ([`gpu-forward-pass/04 §6.4`](../gpu-forward-pass/04-cuda-forward-pipeline.md#64-two-subtle-correctness-points-interview-gold)).

---

## 9. Asynchrony: streams, events, copy engines

```
 host thread:  launch K1 ─ launch K2 ─ memcpyAsync ─ (free to do other work) ─ sync ─▶
 stream S:            [ K1 ][ K2 ][ copy ]                                    ▲ waits here
 copy engine:                       [ DMA ]   ← separate hardware; can overlap kernels in other streams
```

- **A launch is asynchronous:** it enqueues work and returns in microseconds (launch overhead ~a few µs each).
- **One stream is in-order;** different streams *may* overlap.
- **Events** mark points in a stream, for timing (`cudaEventElapsedTime`) and cross-stream dependencies.
- **Pinned host memory** is needed for truly asynchronous copies (module 03 §9; [`gpu-forward-pass/03 §6`](../gpu-forward-pass/03-memory-and-data-movement.md#6-every-transfer-enumerated)).

---

## 10. Numerics on the GPU

| Topic | Fact | SplatScope consequence |
|---|---|---|
| FP64 rate | 1/32 of FP32 on Turing GeForce | Never let `1.0` (a double) into device code (review A9) |
| FMA contraction | nvcc fuses `a*b+c` by default | CPU and GPU differ by a few ulps, so compare with tolerances |
| Fast intrinsics | `__expf`, `__fdividef`; `--use_fast_math` | Faster `exp` in the render inner loop. Measure the accuracy impact against the oracle |
| Half precision | `__half`, `__half2` storage and arithmetic | Storage for SH `f_rest` halves bandwidth ([`gpu-forward-pass/07 §3`](../gpu-forward-pass/07-platform-and-resolved-decisions.md#3-memory-budget-on-a-4-gb-gpu)) |
| Denormals | Flushed to zero with fast math | Tiny alphas become 0, which the α ≥ 1/255 skip already handles |

---

## 11. Reading each SplatScope kernel through this model

Write down predictions like these, then check them with measurements in module 08.

| Kernel | Parallelism | Memory pattern | Predicted bottleneck | Divergence |
|---|---|---|---|---|
| `build_cov3d` (once) | N threads | Coalesced reads and writes | Bandwidth; irrelevant (runs once) | None |
| Preprocess | N threads | Coalesced SoA + per-thread SH blocks | **Bandwidth** (SH dominates at degree 3) | Culling (cheap) |
| Scan (CUB) | N | Streaming | Bandwidth; already near optimal | — |
| Duplicate | N threads, variable work | Scattered writes of varying lengths | Write bandwidth + **load imbalance** | Loop-length divergence |
| Radix sort (CUB) | L | Several streaming passes over 12–24 B/item | **Bandwidth × passes** (fewer `end_bit` bits helps) | — |
| Tile ranges | L threads | Coalesced reads, sparse writes | Trivial | Minimal |
| Render | W·H threads in 256-thread tiles | Gathers into shared memory + broadcast reads; `exp` per pixel×Gaussian | **Mixed:** ALU (`exp`) and latency; **load imbalance across tiles** | Early termination |

---

## 12. Alternatives: GPU programming models

| Model | Pros | Cons | Fit here |
|---|---|---|---|
| **CUDA C++** (chosen) | Best tooling (Nsight, sanitizers); CUB/Thrust; the reference implementation is CUDA | NVIDIA only | ✅ Your GPU, your goals |
| OpenGL / Vulkan **compute shaders** (GLSL) | Portable; no interop needed (same API as presentation); you already know GLSL | Weaker tooling; no CUB (write your own scan and sort, a great exercise but a lot of work) | A good *second* implementation for learning |
| HIP | Nearly CUDA syntax; runs on AMD and NVIDIA | Smaller ecosystem | If portability to AMD matters |
| SYCL | Standard C++, multi-vendor | Toolchain maturity varies | Cross-vendor production code |
| Triton / CuPy / Numba (Python) | Fast iteration; good for prototyping kernels | Less control over shared memory and warp-level details (Triton abstracts them) | Prototyping maths, not this project's goals |
| Slang | Modern shading language with autodiff; targets several backends | Newer ecosystem | Interesting if you later add training (autodiff) |

---

## 13. Apply to SplatScope

**Step 1: startup device report.** Print name, compute capability, SM count, total and free VRAM, L2 size, max threads per SM, and registers per SM (from `cudaGetDeviceProperties`), plus `GL_RENDERER`. Keep it permanently in the log.

**Step 2: occupancy table.** Build with `-Xptxas -v`. For every kernel, record registers, shared memory and spills, and compute theoretical occupancy by hand with the method in §6. Then compare with Nsight Compute's "Occupancy" section.

**Step 3: coalescing micro-benchmark** (a ~50-line standalone `.cu` in `tools/bench/`). Read 256 MB with stride 1, 2, 4, 8, 32 and a random permutation, and plot effective GB/s against stride. You'll *see* §4's table on your own GPU. Then repeat with `float4` loads.

**Step 4: divergence micro-benchmark.** Run a kernel where lanes take a costly branch with probability p ∈ {0, 1/32, 0.5, 1}, and plot time against p. Notice that p = 1/32 already costs almost as much as p = 1.

**Step 5: prediction table.** Copy §11's table into your experiment log (module 08) with *your* predictions, before profiling.

**Checkpoint:**
- You can explain each data point in steps 3–4 with a sentence about sectors or masks.
- Your occupancy-by-hand matches Nsight within one block.

---

## 14. The senior lens

- How many threads does this kernel launch? Enough to fill 16 SMs × 32 warps several times?
- For each load: how many sectors does a warp touch, and how many bytes does it use?
- Which branches vary *per lane*? Which are uniform?
- What limits occupancy: registers, shared memory, or block size? Does it matter for this kernel?
- Where does the host wait for the device, and why?

## 15. Interview probes

| Question | Strong answer |
|---|---|
| What is a warp? | 32 threads executing in lockstep (SIMT); the unit of scheduling |
| How does a GPU hide memory latency? | By switching between many resident warps; this requires enough occupancy or ILP (Little's law) |
| What is coalescing? | A warp's addresses combine into as few 32 B sectors as possible; consecutive aligned access is ideal |
| What causes bank conflicts? | Multiple lanes accessing different addresses in the same shared-memory bank in one instruction |
| Is 100% occupancy always the goal? | No: it's a tool for hiding latency. Compute-bound kernels with ILP can be optimal at lower occupancy |
| Why can't blocks synchronise with each other inside a kernel? | Blocks may not all be resident at once and have no ordering guarantees; use kernel boundaries (or cooperative launch) |
| Divergence cost? | Both paths run serially with masks; cost ≈ the sum of the paths taken by any lane |
| Why is the default stream problematic? | The legacy default stream implicitly synchronises with other blocking streams, serialising work you meant to overlap |
