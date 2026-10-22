# 06 — Data-Oriented Design

> **Mental model:** *the hardware doesn't fetch variables, it fetches **cache lines** (64 B on CPUs, 32 B sectors / 128 B lines on NVIDIA GPUs). Performance is mostly about how many lines you touch, and how many of the bytes in them you actually use. Design your data around how it's **accessed**, and the code becomes simple loops over arrays.*

---

## 1. The problem: the memory wall

A CPU or GPU core can do an arithmetic operation in about a nanosecond or less. A trip to DRAM costs ~100 ns on a CPU (hundreds of cycles) and several hundred cycles on a GPU. Over decades, compute has outgrown memory latency and bandwidth: **the memory wall**.

```
 CPU (typical laptop)              latency       size
 ┌──────────┐
 │ registers│                       ~0           ~1 KB
 ├──────────┤
 │ L1 cache │                       ~1 ns        32–48 KB / core
 ├──────────┤
 │ L2 cache │                       ~4 ns        256 KB–2 MB / core
 ├──────────┤
 │ L3 cache │                       ~10–20 ns    several MB, shared
 ├──────────┤
 │ DRAM     │                       ~80–100 ns   GBs
 └──────────┘   each step down: ~3–10× slower, ~10× bigger
```

**Consequence:** code that does "more work" on well-laid-out data often beats code that does "less work" while chasing pointers. Classic OOP modelling (one heap object per thing, linked by pointers) is the worst case for this hardware.

---

## 2. First principles

### 2.1 Cache lines and spatial locality

When you read one `float`, the hardware brings in the whole 64-byte line around it. If the next 15 floats you need are in that line, they're free. If you only needed 4 bytes of it, you paid for 64. Track **bytes used ÷ bytes fetched** (the *utilisation*).

### 2.2 Temporal locality

Reusing data while it's still in cache is free. Blocking or tiling an algorithm (processing a tile of the image while its Gaussians are hot) is how you get temporal locality. The 3DGS render kernel's shared-memory batch is this idea, made explicit on the GPU.

### 2.3 Prefetching and linear access

Hardware prefetchers detect sequential and strided access and fetch ahead. **Linear walks over arrays are the fastest access pattern on any hardware.** Random gathers (following indices or pointers) defeat the prefetcher.

### 2.4 Alignment

A 16-byte value that straddles two cache lines costs two fetches. SIMD loads (CPU) and vector loads (GPU `LDG.128`) need natural alignment. Use `alignas(16)` for `Vec4f`, and remember that `std::vector` respects over-alignment since C++17 (aligned `new`).

---

## 3. Layouts: AoS, SoA, AoSoA

```
 AoS — Array of Structures          struct G { Vec3 mean; float opacity; Vec3 scale; Vec4 rot; float sh[48]; };
 ┌──────────────────────────────────────┬──────────────────────────────────────┬───
 │ mean op scale rot sh[48]…  (≈236 B)  │ mean op scale rot sh[48]…            │ …
 └──────────────────────────────────────┴──────────────────────────────────────┴───
   A loop that only needs `mean`: uses 12 of every ~236 bytes → ~5 % utilisation ✗

 SoA — Structure of Arrays          struct Gs { vector<Vec4f> mean_op; vector<Vec4f> cov_a; vector<float> sh; … };
 mean_op │ G0 │ G1 │ G2 │ G3 │ G4 │ … contiguous, 100 % utilisation for a mean-only loop ✓
 cov_a   │ G0 │ G1 │ G2 │ G3 │ G4 │ …
 sh      │ G0 (192 B) │ G1 (192 B) │ …

 AoSoA — blocks of K (e.g. K = 8 for AVX, 32 for a warp), SoA inside each block
 ┌ block 0 ────────────────────────────┐┌ block 1 ──────────────────────────┐
 │ x0..x7 │ y0..y7 │ z0..z7 │ op0..op7 ││ x8..x15 │ y8..y15 │ …            │
 └─────────────────────────────────────┘└───────────────────────────────────┘
   SIMD-width-friendly AND keeps one entity's fields within a few lines
```

| Layout | Best when | Worst when | Notes |
|---|---|---|---|
| **AoS** | You process *all fields of one entity together*, and access is random (one entity at a time) | Loops that touch a few fields of many entities | Natural to write; the default in OOP |
| **SoA** | Loops over many entities touching few fields; SIMD/SIMT | Random access to all fields of one entity (one line per field) | The GPU default; what kernels want |
| **AoSoA** | SIMD on CPU with mixed access | Code complexity (index = block·K + lane) | Used in high-performance CPU code (physics, ray tracing) |
| **Hybrid: hot/cold split** | A few fields are hot (used every frame), many are cold (rarely used) | — | Keep hot fields packed; move cold ones (names, debug info, the loader's source line) elsewhere |

### 3.1 The same choice, seen per kernel in SplatScope

| Stage | Access pattern | Best layout |
|---|---|---|
| Loader | Entity at a time, all fields (convert one Gaussian) | Either; SoA output (that's what the GPU wants) |
| Preprocess | Thread *i* reads Gaussian *i* (several fields, coalesced across the warp) | **SoA**, with 16 B fields |
| SH evaluation | Thread *i* reads 48 floats of Gaussian *i* | Per-Gaussian contiguous block (AoS within the SH array) or coefficient-major SoA ([`07 alternatives`](../gpu-forward-pass/07-platform-and-resolved-decisions.md#alternatives--trade-offs)) |
| Render gather | Random by sorted id; needs xy + conic + rgb together | **Packed AoS "render packet"** may win here: an experiment for module 08 |

**Lesson:** "SoA vs AoS" isn't a religion. It's a **per-access-pattern** decision, and one system can use both: SoA for storage, a packed AoS record where a gather needs it.

---

## 4. Other data-oriented techniques

### 4.1 Hot/cold splitting

`HostScene` has hot data (per-frame on the GPU) and cold data (`SceneInfo`, source file name, bounds). Keep cold data out of hot arrays. On the GPU, the "cold" data is everything not needed by the current kernel: that's why there are many narrow SoA arrays instead of one wide struct.

### 4.2 Existence-based processing (instead of flags)

Instead of a `bool visible` per Gaussian and `if (visible)` in every later loop, **produce a compacted list** of visible ids, or encode existence in the data (`tiles_touched == 0` means it contributes nothing to the scan or to duplication). Later stages then process only what exists. Stream compaction on the GPU is an optimisation candidate for scenes where most Gaussians are culled.

### 4.3 Sort for locality

The radix sort in the pipeline isn't only for correctness (depth order). It also *groups* each tile's work contiguously. Sorting data so that what's processed together is stored together is a general DOD technique (Morton or Z-order sorting of the scene by position is another, see module 11).

### 4.4 Batch-sized interfaces

APIs that take **one** item (`project(const Gaussian&)`) force per-item call overhead and hide the batch from the optimiser. APIs that take **arrays** (`project_all(span<const …>, …)`) let the implementation vectorise, parallelise or launch a kernel. SplatScope's `IRenderer::render` takes the whole frame; its kernels take whole arrays.

### 4.5 False sharing (CPU multithreading)

Two threads writing *different* variables that share a cache line force that line to bounce between cores, which can be slower than a single thread. If you parallelise the CPU oracle over rows (OpenMP), make sure per-thread accumulators are padded to 64 B (`alignas(std::hardware_destructive_interference_size)`, C++17), or simply write to disjoint image rows.

---

## 5. OOP vs DOD isn't either/or

| Concern | Use OOP-style abstractions | Use DOD |
|---|---|---|
| **Architecture boundaries** (renderer, presenter, loader) | ✅ interfaces, RAII owners, encapsulation | — |
| **Bulk data** (N Gaussians, L instances, pixels) | ❌ no per-item objects or virtuals | ✅ SoA arrays + functions or kernels |
| **Configuration and state** (camera, settings) | ✅ small value types | — |

**Rule of thumb:** *objects for the few, arrays for the many.* There are about 10 objects in SplatScope's architecture and millions of array elements.

---

## 6. Alternatives

| Choice | Option A | Option B | Trade-off |
|---|---|---|---|
| Host scene storage | **SoA `std::vector`s in GPU layout** (chosen) | AoS `vector<Gaussian>` converted at upload | B is nicer to construct in the loader but costs a conversion pass and a second copy in memory (bad for a 1.5 GB scene) |
| Scene memory | Separate vectors per field | One big allocation with offsets (an arena) | The arena gives one allocation and one free, and it's better for mmap/streaming. Separate vectors are simpler and safer (each has a size) |
| Per-Gaussian render data | SoA (`mean2d`, `conic_opacity`, `rgb` arrays) | Packed 32 B `RenderPacket` | Measure sectors/request in the render gather (module 08) |
| CPU oracle loops | Plain loops + `-O2` | Explicit SIMD (intrinsics) or ISPC | The oracle values clarity over speed. Use OpenMP over rows if it's too slow |

---

## 7. Apply to SplatScope

**Step 1:** redesign `HostScene` as SoA in GPU layout (signature in [`gpu-forward-pass/05 §5`](../gpu-forward-pass/05-cpp-implementation-guide.md#5-host-side-scene-types)). Write its invariants: equal lengths, finite values, normalised `QuatWXYZ`, `sh.size() == N·48`.

**Step 2:** in the loader, convert **property group by property group** directly into the SoA arrays, releasing each tinyply buffer as you go (keeps peak host RAM low, [`07 §3.4`](../gpu-forward-pass/07-platform-and-resolved-decisions.md#34-host-ram-the-loader-is-the-peak)). Filter invalid Gaussians with an in-place **stable compaction** that moves all arrays together (one pass, no second cloud).

**Step 3:** write the CPU oracle's preprocess as a loop over arrays that calls the shared `SPLAT_HD` functions, with output in SoA arrays shaped exactly like the GPU's `GeometryBuffers`. Stage-by-stage comparison is then just comparing arrays.

**Step 4: a micro-experiment that sticks.** Time a loop that reads only the means, over (a) `vector<GaussianAoS>` with all 62 floats, and (b) `vector<Vec4f>`, for N = 6M. Predict the ratio from bytes fetched *before* running it. Then run it with `perf stat -e cache-misses,cache-references`. Writing the prediction down first is the habit this step builds.

**Checkpoint:**
- Loader peak RSS (`/usr/bin/time -v`) is under 2× the file size for a large scene.
- The micro-experiment result is within 2× of your prediction, with an explanation if it isn't.

---

## 8. The senior lens

- For each hot loop: what are its access patterns, and how many bytes does it fetch vs use?
- Is there per-item dispatch (virtual call, pointer chase) inside a loop over millions of items?
- Which fields are hot and which are cold? Are they separated?
- Could later stages process only what exists, instead of checking flags?
- Is the interface batch-shaped, or does it force item-at-a-time calls?

## 9. Interview probes

| Question | Strong answer |
|---|---|
| AoS vs SoA? | It depends on access: SoA for many-entities-few-fields loops and SIMD/SIMT; AoS for whole-entity random access. Hybrids (AoSoA, hot/cold) exist |
| What is false sharing? | Different threads writing different data on the same cache line, so the line ping-pongs between cores. Fix with padding or disjoint writes |
| Why are linked lists slow? | Each node is a dependent cache miss; the prefetcher can't predict the next address |
| What's a cache line, and why does it matter? | The unit of transfer (64 B on CPU). Utilisation = used bytes ÷ fetched bytes drives effective bandwidth |
| How would you speed up a loop over millions of objects with virtual methods? | Group by type (or remove the virtual dispatch), convert to SoA arrays, and process each type's array in a tight loop |
