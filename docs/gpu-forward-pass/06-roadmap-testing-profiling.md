# 06 — Roadmap, Testing & Profiling

> **Mental model:** *bring up a GPU pipeline one verified stage at a time, each checked against an oracle, with a visual debug view for every intermediate buffer.* A GPU bug that crosses two stages before you look is often ten times harder to find. The milestones below are ordered so that every stage's input has already been verified when you start it.

---

## 1. Milestones

Each milestone has a **learning goal**, the **work**, and a **definition of done** (DoD) that can be checked mechanically.

### M0 — Foundations (no new features)

- **Learning goal:** RAII, ownership, build hygiene.
- **Work:**
  - CMake restructure ([05 §8](05-cpp-implementation-guide.md#8-cmake-structure-target-graph-not-the-full-file)) with FetchContent, so a fresh clone builds. CPU-only build option.
  - `DeviceBuffer<T>`, `CudaStream`, `CudaEvent`, `PinnedBuffer`, `GlHandle`, throwing `CUDA_CHECK`.
  - `Window` non-copyable/non-movable; `GlfwRuntime`; `Presenter` extracted out of the renderers; `IRenderer::render(FrameRequest, ImageView)`.
  - `HostScene` SoA from the loader (no Eigen in the stored scene); `shared_ptr<const HostScene>`.
  - Fix the conventions: $\det R = +1$ and the one texcoord flip in the presenter (review B1).
  - Add the test framework and a `tests/` target.
- **DoD:**
  - `git clone && cmake --preset release && cmake --build` works on a clean machine.
  - Builds with `SPLATSCOPE_WITH_CUDA=OFF`.
  - Unit test: `det(R) == 1 ± 1e-5` over random orbit parameters.
  - ASan/UBSan run of the CPU path is clean.

### M1 — Shared maths + a correct CPU oracle

- **Learning goal:** EWA projection, conic, the compositing equation, front-to-back vs back-to-front.
- **Work:**
  - The `core/` `SPLAT_HD` header ([05 §4](05-cpp-implementation-guide.md#4-shared-math-for-host-and-device)).
  - Rewrite `CpuRenderer` as preprocess (shared functions) → global depth sort → per-pixel front-to-back compositing in a **float** accumulation buffer, all using one `RenderSettings`. This fixes review E1–E5.
- **DoD:** the analytic unit tests in §2.1 pass, and the CPU render of a real scene looks right at all orbit angles (no square clipping, no darkening).

### M2 — GPU preprocess

- **Learning goal:** one-thread-per-element kernels, coalescing, "every slot written".
- **Work:** the `build_cov3d` one-shot kernel; `preprocess` per [04 §1](04-cuda-forward-pipeline.md#1-preprocess-one-thread-per-gaussian) (fixes A2, A5, A6, A7, A8, A9). Debug views `Centers` and `Ellipses`.
- **DoD:**
  - Per-Gaussian outputs (`mean2d, depth, radius, conic, tiles_touched`) match the CPU oracle within tolerance for every Gaussian, at 10 fixed camera poses.
  - `compute-sanitizer --tool initcheck` and `--tool memcheck` are clean.

### M3 — Binning: scan, duplicate, sort, ranges

- **Learning goal:** prefix sum as allocation, key packing, radix sort, segment boundaries.
- **Work:** [04 §2–5](04-cuda-forward-pipeline.md#2-scan-tiles_touched--offsets) (fixes A1, A3, A4); grow-only binning buffers; pinned L readback on the explicit stream; `GaussiansPerTile` heatmap.
- **DoD:**
  - Host-side checks in Debug: keys sorted, ranges partition `[0, L)`, and each tile's depth sequence is non-decreasing.
  - Device `assert(off == offsets[idx])` never fires.
  - L matches the CPU-computed L exactly. It's integer logic, so tolerance 0.
  - No `cudaMalloc`/`cudaFree` in steady state (verify in the Nsight Systems timeline).

### M4 — Render kernel

- **Learning goal:** block = tile, shared-memory staging, barriers, early termination.
- **Work:**
  - First write a **naïve** version: each thread loops over its tile's range reading global memory directly. Get it correct.
  - Then write the shared-memory batched version per [04 §6](04-cuda-forward-pipeline.md#6-render-one-block-per-tile-one-thread-per-pixel). Keep both behind a flag and compare their speeds.
  - Debug views: `ContribCount`, `Depth`, `Alpha`, `CpuGpuDiff`.
- **DoD:**
  - GPU vs CPU image: PSNR ≥ 45 dB and max |Δ| ≤ 2/255 on the golden poses.
  - `racecheck` and `synccheck` are clean.
  - Correct at 801×601.

### M5 — Presentation and measurement

- **Learning goal:** data movement, pinned memory, streams, interop.
- **Work:**
  - `StageTimer` and an on-screen/console HUD showing N, N_visible, L, L/N_visible and per-stage ms.
  - Phase A with pinned staging, then Phase B PBO interop with runtime fallback.
  - Resize handling. `glfwSwapInterval(0)` when measuring.
- **DoD:**
  - The Nsight Systems timeline shows exactly the sync points listed in [03 §5](03-memory-and-data-movement.md#5-streams-and-synchronisation).
  - With interop on, no 8 MB D2H/H2D copies appear.
  - Resize works, including minimise/restore.

### M6 — Extensions (pick by interest)

- **SH degree 1–3**, with view-dependent colour. Store `half` later if memory matters.
- **Opacity-aware radius** ([04 §1.5](04-cuda-forward-pipeline.md#15-dilation-conic-radius)): measure the change in L.
- **Packed `RenderPacket` AoS** for the render gather ([03 §3](03-memory-and-data-movement.md#soa-vs-aos-decided-per-kernel)): measure sectors/request.
- **Tighter tile culling:** test each tile against the actual ellipse instead of the bounding circle. Measure L again.
- **Load a COLMAP camera** from the training data, render the view and compare with the training image. This is the ultimate end-to-end check of your conventions.

---

## 2. Testing strategy

### 2.1 The test pyramid

```
              ┌───────────────────────────────┐
              │ Golden images (few, slow)     │  CPU vs GPU, fixed poses, PSNR / max|Δ|
              ├───────────────────────────────┤
              │ Stage tests (GPU vs CPU)      │  per-Gaussian outputs; L; key order; ranges
              ├───────────────────────────────┤
              │ Analytic unit tests (host)    │  SPLAT_HD maths with closed-form answers
              └───────────────────────────────┘
        + compute-sanitizer (memcheck / initcheck / racecheck / synccheck) on every milestone
```

### 2.2 Analytic unit tests (fast, host-only, exact answers)

| Test | Expected |
|---|---|
| Identity rotation, scale $s$ | $\Sigma = \mathrm{diag}(s^2)$ |
| Quaternion $(w,x,y,z)$ and $(-w,-x,-y,-z)$ | identical Σ (double cover) |
| Isotropic Gaussian ($\sigma$) on the optical axis at depth $z$ | $\Sigma_{2D} = (f\sigma/z)^2 I$ exactly (+0.3 after dilation) |
| Same Gaussian off-axis | $\Sigma_{2D}$ elongated **radially**; compare with a finite-difference Jacobian (this is the A5 regression test) |
| Conic × Σ₂D | $\approx I$ |
| `get_tile_rect` with centre on a tile boundary, radius 0, radius bigger than the screen, centre off-screen with overlap | hand-computed rectangles; `tile_count` ≥ 0 always |
| Depth bit packing | for random positive floats `a < b` ⇒ `bits(a) < bits(b)` |
| Compositing of 2 Gaussians, front-to-back vs back-to-front | equal within 1e-6 |
| `det(R)` of the orbit camera | $+1$ |

### 2.3 Synthetic scenes (tiny PLYs or in-code `HostScene`s)

Each one targets a specific failure mode:

1. **Empty scene (N = 0):** no launches, no crash, background colour.
2. **N = 1, centred, isotropic:** check the radius and tile count by hand.
3. **N = 257:** catches grid-size off-by-one errors (one more than a block).
4. **Two overlapping Gaussians at depths 3.2 and 3.7, same tile:** catches A3 (value-cast depth).
5. **A Gaussian straddling a tile corner:** exactly 4 instances.
6. **A Gaussian centred off-screen, overlapping the image:** catches A7.
7. **A Gaussian behind the camera, and one at z = 0.1:** culled.
8. **One huge Gaussian covering the whole screen:** L = number of tiles; stress test.
9. **A dense random cloud (100k):** for golden images and timing.
10. **Window sizes 800×600, 801×601, 16×16, 1×1:** edge tiles and barrier correctness.

### 2.4 Tolerances

The host and device won't be bit-identical (FMA contraction, `expf` vs `__expf`; see [05 §4](05-cpp-implementation-guide.md#4-shared-math-for-host-and-device)). Compare in two ways:

- **Per-Gaussian floats:** relative tolerance ~1e-4.
- **Integers** (`radius`, `tiles_touched`, L): exact. If they differ, it's a real bug, or a float sitting exactly on a `ceil` boundary. Log those cases instead of loosening the tolerance.
- **Images:** PSNR ≥ 45 dB and max |Δ| ≤ 2/255. A few-pixel max error at silhouette edges is expected when a Gaussian's α sits right at the 1/255 threshold.

---

## 3. Profiling

### 3.1 Always-on instrumentation

`StageTimer` with CUDA events around preprocess / scan / duplicate / sort / ranges / render / present. It reports through `FrameStats`, with no host syncs mid-frame. Log or show:

- N, N_visible, L, L/N_visible
- per-stage ms, total GPU ms, CPU frame ms

**Correlate them:** sort time should track L; render time should track `GaussiansPerTile` imbalance and `ContribCount`.

### 3.2 Nsight Systems: "where does the time go between kernels?"

Run with `nsys profile -t cuda,opengl,osrt ./splatscope scene.ply` and look for:

- Gaps on the GPU timeline, which mean host-side stalls. Expect one tiny gap at the L readback.
- Any `cudaMalloc`/`cudaFree` in steady state; there should be none after M3.
- 8 MB memcpys per frame; there should be none with interop.
- `cudaDeviceSynchronize` anywhere; there should be none.
- Implicit syncs from default-stream use.

### 3.3 Nsight Compute: "why is this kernel slow?"

Focus on `render` and `preprocess` (`ncu --set full -k render_tiles ./splatscope …`):

- **Render:**
  - Memory throughput vs compute throughput.
  - Warp divergence: threads in a tile that terminate early idle until the whole block votes done.
  - Shared-memory bank conflicts.
  - Occupancy limited by registers or shared memory.
  - *Sectors/request* on the gather loads (the SoA vs AoS experiment).
- **Preprocess:** confirm `LDG.128` for `mean_opacity` and `cov3d_a` (view the SASS); check registers/spills (`-Xptxas -v`).

### 3.4 Rough expectations (to calibrate, not to target)

The reference CUDA rasterizer renders million-Gaussian scenes at 1080p in a few milliseconds on recent desktop GPUs, and the render and sort stages dominate. For a learning implementation, aim to be **correct and within a small factor of that**. The more valuable deliverable is being able to *explain* every millisecond on the timeline.

---

## 4. Interview probes this project prepares you for

Short answers; the linked sections have the full reasoning.

| Question | One-line answer |
|---|---|
| Why tiles instead of sorting per pixel? | One global radix sort of (tile, depth) keys gives every tile a contiguous depth-sorted list. Per-pixel sorting is far more work and memory ([04](04-cuda-forward-pipeline.md)) |
| Why front-to-back? | Same result as back-to-front, but transmittance only decreases, so threads can stop once T < 1e-4 |
| Why is the 2D covariance $J W\Sigma W^\top J^\top$? | A linear map $A$ sends $\mathcal N(\mu,\Sigma)$ to $\mathcal N(A\mu, A\Sigma A^\top)$; perspective is linearised by its Jacobian at the mean |
| What does +0.3 on the diagonal do? | Convolves with a ~pixel-sized Gaussian: anti-aliasing plus conditioning. Convolution *adds* covariances |
| Why prefix sum before duplication? | Parallel allocation: each Gaussian learns its private output offset deterministically, without atomics |
| How can a float be part of an integer sort key? | For non-negative IEEE-754 floats, the bit pattern ordered as unsigned matches the value order |
| Why is there a device→host copy every frame? | The radix sort's item count (L) is a host parameter, and the binning capacity is ensured on the host |
| Why can't edge-tile threads just `return`? | They'd skip `__syncthreads`, giving undefined behaviour or a deadlock. They must keep loading shared memory and simply not write output |
| What happens after an illegal memory access in a kernel? | The error is asynchronous and *sticky*: it is reported at the next sync, and the context is unusable afterwards |
| Pinned vs pageable memory? | DMA needs page-locked physical memory. Pageable copies bounce through a driver buffer and are synchronous |
| Why can't `cudaFree` be in the frame loop? | It synchronises the whole device |
| Why does `unique_ptr<Impl>` PIMPL need an out-of-line destructor? | `default_delete` needs the complete type at the point of destruction |
| Why must a moved-from owner null its pointer? | Its destructor still runs |
| What breaks if `Window` is moved? | GLFW's user pointer still points at the old `this` |
| SoA or AoS? | It depends on the access pattern. Coalesced per-element access favours SoA; a random gather of several fields favours a packed AoS record |
| Why doesn't a `float3` array get 128-bit loads? | It is 12-byte aligned, not 16, so the compiler issues three 32-bit loads. Coalescing is fine; it's the instruction count that suffers |
| Why call `cudaGraphicsUnmapResources` before GL draws? | Unmapping orders all prior CUDA work on that stream before subsequent GL work. Without it there's a race |
