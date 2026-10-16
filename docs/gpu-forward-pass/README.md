# SplatScope — GPU Forward Pass: Review & Architecture Plan

This folder reviews the `gpu_renderer` branch and lays out how to finish the GPU forward pass. It covers the architecture, ownership, memory and data movement, and the CUDA pipeline, from the module diagram down to the class signatures.

**Baseline reviewed:** branch `gpu_renderer` @ `12208c2` (PR #1), diffed against `main` @ `9e95435`. All `file:line` references in these documents point to that commit.

**Scope (deliberately narrow):** SplatScope is a learning-oriented viewer. It has a forward pass only: no training, no backward pass and no densification. The goal is to understand 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) and the C++/CUDA engineering around it. "Production-ready" here means *correct, leak-free, debuggable, measurable, and structured so each concept lives in one place*. It does not mean squeezing out the last 10% of FPS.

---

## The one-sentence mental model

> **Project every 3D Gaussian to a 2D ellipse. Bucket the ellipses into 16×16 screen tiles with one global sort keyed by `(tile, depth)`. Each tile then becomes an independent front-to-back alpha-compositing problem, and one thread block solves it.**

Everything in the CUDA pipeline exists to make that sentence parallel:

```
            per-Gaussian (N threads)            per-instance (L)          per-tile / per-pixel
 ┌──────────────┐   ┌────────────┐   ┌────────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1 preprocess │──▶│ 2 scan     │──▶│ 3 duplicate    │──▶│ 4 radix  │──▶│ 5 tile       │──▶│ 6 render     │
 │ project,     │   │ tiles_     │   │ emit (key,id)  │   │ sort by  │   │ ranges       │   │ block=tile,  │
 │ cov2D, conic,│   │ touched →  │   │ key = tile<<32 │   │ key      │   │ [start,end)  │   │ thread=pixel │
 │ radius, rect │   │ offsets    │   │     | depth    │   │          │   │ per tile     │   │ front→back α │
 └──────────────┘   └────────────┘   └────────────────┘   └──────────┘   └──────────────┘   └──────────────┘
        N                 N          ── host reads L ──▶       L                L                 W×H
```

N = number of Gaussians. L = number of (Gaussian, tile) *instances*, which changes with the view and is known only after step 2. That one host-side readback of L is the only CPU↔GPU sync point the algorithm itself requires.

---

## Documents (read in this order)

Each document ends with an **Alternatives & trade-offs** section: the other designs considered, how they compare, and when you would switch.

| # | Document | What it answers |
|---|---|---|
| 01 | [Code review](01-code-review.md) | What is wrong or risky in the current branch, with severity, the failure mode and the fix |
| 02 | [Architecture](02-architecture.md) | Module/class split, ownership graph, construction/destruction order, per-frame control flow, conventions, error policy |
| 03 | [Memory & data movement](03-memory-and-data-movement.md) | Buffer lifetime classes, allocation policies, layouts (SoA/float4), the memory budget, every host↔device transfer, streams, CUDA-GL interop |
| 04 | [CUDA forward pipeline](04-cuda-forward-pipeline.md) | Kernel-by-kernel spec with the math (EWA projection, conic, tile rect, key packing, compositing), invariants and debug views |
| 05 | [C++ implementation guide](05-cpp-implementation-guide.md) | Signature-level sketches: RAII wrappers, owner vs view types, PIMPL, shared `__host__ __device__` math, CMake target layout |
| 06 | [Roadmap, testing & profiling](06-roadmap-testing-profiling.md) | Milestones with definition-of-done, the test pyramid, synthetic test scenes, profiling, interview probes |

---

## TL;DR — the eight things to fix first

These are the issues that currently prevent a correct image. Details and fixes are in [01](01-code-review.md).

1. **Duplicate kernel guard is inverted.** `if (d_radii[idx] > 0) return;` skips every *visible* Gaussian (`gaussian_pipeline.cu:34`).
2. **Per-frame outputs are never initialised for culled Gaussians.** `radii` and `tiles_touched` keep stale or garbage values, so the prefix sum and L are garbage (`gaussian_pipeline.cu:89-97`).
3. **Depth is packed as `(uint32_t)depth`.** This truncates to whole units, so everything within one world unit of depth ties. You want the IEEE-754 *bit pattern* (`__float_as_uint`) (`gaussian_pipeline.cu:56`).
4. **The tile rectangle is computed two different ways.** The count pass divides by the tile *size* (16) and clamps; the write pass divides by the tile *count* (`tile_grid.x`) and doesn't clamp. When the counts disagree, the writes go out of bounds (`gaussian_pipeline.cu:42-45` vs `:131-134`).
5. **Covariance projection is transposed.** The code computes $J^\top \Sigma' J$ instead of $J \Sigma' J^\top$. I checked this numerically: the 2×2 results differ whenever the Gaussian is off-axis (`gaussian_math.cuh:76`).
6. **The low-pass dilation is multiplied, not added.** `cov2d.x * 0.3f` should be `cov2d.x + 0.3f`. The eigenvalues also use the un-dilated matrix while `det` uses the dilated one (`gaussian_pipeline.cu:110-125`).
7. **The view matrix is a reflection (det = −1).** The image only looks right because OpenGL's bottom-up texture rows flip it back (`camera.cpp:124-126`).
8. **Binning buffers are `cudaMalloc`/`cudaFree`d every frame.** `free_scene` also leaks `d_cov3d`, and `free_scratch` leaks the CUB temp buffer.

---

## Whole-system alternatives (before committing to the plan)

There are three credible ways to build a Gaussian-splatting viewer. They differ in *where the sorting happens* and *who does the blending*: your own compute kernels, or the GPU's fixed-function graphics pipeline.

| | **A. CUDA tile rasterizer + GL present** (this plan) | **B. Graphics-pipeline splatting** (GL/Vulkan instanced quads) | **C. Ray-traced Gaussians** (OptiX/Vulkan RT) |
|---|---|---|---|
| How it works | Preprocess → tile/depth keys → radix sort → one block per tile composites front-to-back in a compute kernel | Sort Gaussians by depth (compute shader or CPU) → draw one camera-facing quad per Gaussian → the fragment shader evaluates the 2D Gaussian → fixed-function blending, back-to-front | Proxy geometry per Gaussian in a BVH; each ray collects hits in depth order and composites |
| Ordering correctness | Per-tile, by centre depth (same approximation as the paper) | Global, by centre depth (same approximation) | Per-ray hit order (more correct, no popping) |
| Early termination | Yes (T < 1e-4) | No: every covered fragment is blended (overdraw) | Yes |
| What you learn | CUDA memory model, scans, radix sort, shared memory, barriers, interop | Graphics pipeline, blending equations, instancing, GLSL, compute-shader sorting | BVHs, RT pipelines, any-hit shaders |
| Matches the paper's rasterizer | Exactly (images comparable to training renders) | Close; differs in blending precision and the per-tile vs global sort | Different algorithm |
| Portability | NVIDIA only | Any GPU, even WebGL (this is how most web viewers work) | RT-capable GPUs; very complex |
| Amount of code | Medium | Low–medium | High |
| Fits the current codebase | Yes, it continues the `gpu_renderer` branch | Would replace the CUDA work | No |

**Recommendation: A.** It's what you've started, it matches the reference algorithm exactly (so you can compare against training renders), and it teaches the CUDA and C++ ownership material you asked for. **B is the best *second* backend:** it reuses the same `HostScene`, `Camera` and `Presenter`. Implementing it after A makes the trade-off between fixed-function blending and a compute rasterizer concrete, which is a strong interview story. C is out of scope for a learning viewer.

Every document below ends with an **Alternatives & trade-offs** section for its own decisions.

---

## Decisions this plan assumes (and open questions)

The plan picks a default for each of these so it is complete as written. If you disagree with one, only the sections listed change.

| Decision | Default assumed | Affects |
|---|---|---|
| Keep the CPU renderer? | **Yes, as the correctness oracle** (fixed to the same math model) | 01-E, 05, 06 |
| Presentation path | **Phase A:** device→host readback (simple, debuggable, can dump PNGs). **Phase B:** CUDA-GL PBO interop (zero PCIe traffic) | 02, 03 |
| Color model | **SH degree 0 (DC) first**; degree-3 SH as a later milestone | 03, 04, 06 |
| Math library on device | **A small in-house `SPLAT_HD` math header** shared by CPU oracle and kernels. Eigen stays at the app layer; glm is dropped | 05 |
| Dependencies | **CMake `FetchContent` with pinned tags** (glm/tinyply are currently missing from the repo) | 01-F, 05 |
| Target | **Single NVIDIA GPU, CUDA ≥ 12, one window, single-threaded render loop** | 02, 03 |

**Questions for you:**

1. Which GPU/driver/CUDA version are you on, and is it a **laptop with hybrid graphics** (Optimus)? If OpenGL runs on the iGPU and CUDA on the dGPU, interop (Phase B) either fails or silently goes through host memory. If so, we stay on Phase A.
2. Do you want a **runtime CPU↔GPU toggle and debug-view hotkeys** (strongly recommended for learning)? If yes, is Dear ImGui acceptable as a dependency, or should it stay keyboard-only?
3. Do you want to go up to **SH degree 3** (view-dependent color)? It roughly quadruples per-Gaussian scene memory (see [03](03-memory-and-data-movement.md#4-memory-budget)).
4. Should the scenes you test with come from the **official 3DGS training output** (`point_cloud.ply` with `f_rest_*`)? Or other exporters too, which may store SH in a different order or quantised?
