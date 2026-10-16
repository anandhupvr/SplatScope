# 03 — Memory Policy & Data Movement

> **Mental model:** classify every buffer by *what its size depends on* (N, L, pixels, or a constant) and *when that changes*. That classification alone decides the allocation policy, the owner and the reallocation trigger. Most GPU memory bugs and stalls come from a buffer being handled with the wrong class's policy. One example: today's binning buffers are L-sized, but they are treated as if they were scoped to a single function call.

---

## 1. Lifetime classes

| Class | Size ∝ | Changes when | Policy | Owner (in `GpuRenderer::Impl`) |
|---|---|---|---|---|
| **Scene** (immutable) | N | new scene loaded | Allocate exactly once, upload, then **const** forever | `SceneBuffers scene_` |
| **Per-Gaussian, per-frame** | N | new scene loaded | Allocate with the scene; **every slot overwritten every frame** | `GeometryBuffers geom_` |
| **Per-instance (binning)** | **L**, view-dependent | **every frame** | **Grow-only** capacity, geometric growth (×1.5), never shrink in steady state | `BinningBuffers bin_` |
| **Per-image** | W·H pixels, ⌈W/16⌉·⌈H/16⌉ tiles | window resize | Reallocate exactly on resize | `ImageBuffers img_` |
| **Per-frame uniforms** | constant (~100 B) | every frame | **No buffer.** Pass `CameraParams` by value as a kernel argument | — |
| **Library temp** (CUB) | f(N) for scan, f(L) for sort | when its client's capacity grows | Query size → grow-only; lives *with the buffer it serves* | inside `geom_` / `bin_` |
| **Host staging** | 4 B (L) / W·H·4 (Phase A image) | resize | **Pinned** (`cudaMallocHost`), grow-only | `PinnedBuffer<T>` members |

**The invariant that makes "overwrite every frame" safe:** a per-frame buffer is either fully written by the stage that produces it (preprocess writes every one of the N slots) or explicitly cleared at the start of the frame (`tile_ranges` with `cudaMemsetAsync`). There is never a third state. Review findings A2 and A1 are both violations of this.

---

## 2. Allocation policy per class

### Why not `cudaMalloc` per frame?

`cudaMalloc` is a driver call that may take a lock and update page tables. `cudaFree` **synchronises the entire device**, so it waits for *all* outstanding work on *all* streams. Four of each per frame (today's code) inserts unpredictable stalls and makes every timing measurement noise.

### The options, and which to use

| Strategy | How | Pros | Cons | Verdict |
|---|---|---|---|---|
| Per-frame malloc/free | today | trivial | stalls, fragmentation | ❌ |
| **Grow-only typed buffers** | `DeviceBuffer<T>::ensure_capacity(n)`: if `n > cap`, free and allocate `max(n, 1.5·cap)`, *discarding contents* | explicit, typed; **each sub-buffer is a separate allocation, so `compute-sanitizer memcheck` catches overruns** | one free/alloc per growth (rare) | ✅ **Use this** |
| Single arena, carved | one `char*` chunk; `obtain(chunk, ptr, count, align)` bumps a cursor (the reference rasterizer does this, because PyTorch owns its memory) | one allocation; fits a framework allocator | untyped; **an overrun from sub-buffer A into B is invisible to memcheck**, since it's all one allocation | learn it, don't use it here |
| Stream-ordered allocator | `cudaMallocAsync`/`cudaFreeAsync` + a memory pool with a high `cudaMemPoolAttrReleaseThreshold` | per-frame alloc becomes cheap and stream-ordered | less explicit; pool tuning | production alternative |

**Retention hook:** the arena's main *advantage* (one allocation) is also why bugs like review A4, where the write pass overflows into the next Gaussian's slots, go undetected. With separate allocations, the last Gaussian's overflow runs past the end of the buffer, and `memcheck` reports that write with the kernel name and thread index.

### Growth details that matter

- **Grow-only buffers don't preserve contents.** Binning buffers are fully rewritten every frame, so `ensure_capacity` frees and then allocates. It never copies. Name it accordingly (`ensure_capacity_discard` if you want it unmistakable).
- **Re-query CUB temp storage on growth.** `SortPairs` temp size depends on `num_items` and `end_bit`. Query it with the *capacity* (not L) so that one temp buffer serves every L up to the capacity.
- **Use `cub::DoubleBuffer` for the sort.** The plain `SortPairs(keys_in, keys_out, vals_in, vals_out, …)` overload allocates an alternate key/value buffer *inside its temp storage*, which costs ~12 bytes more per instance. The `DoubleBuffer` overload ping-pongs between your two buffers and needs only a small temp. The catch: the result may end up in either buffer, so read `d_keys.Current()` / `d_vals.Current()` afterwards. (Interview follow-up: "why can't radix sort be in-place?" Each LSD pass is a stable scatter, and a scatter needs a destination distinct from its source.)
- **Budget cap.** Decide a maximum L (e.g. from `cudaMemGetInfo` at startup). If a frame needs more (the camera is inside a dense region), report it in `FrameStats` and render a degraded frame instead of OOM-aborting.

---

## 3. Layouts

### Scene buffers (what the loader produces, what gets uploaded)

The loader emits **SoA arrays already in device layout**, so upload is a plain `memcpy` with no `reinterpret_cast` of Eigen types (review C6).

| Buffer | Type | Bytes/Gaussian | Content |
|---|---|---|---|
| `mean_opacity` | `float4` | 16 | `xyz` = world mean, `w` = activated opacity (the padding lane put to use) |
| `cov3d_a` | `float4` | 16 | Σ entries `(c00, c01, c02, c11)` |
| `cov3d_b` | `float2` | 8 | Σ entries `(c12, c22)` |
| `rgb` (SH0) | `float4` | 16 | `SH_C0·f_dc + 0.5`, clamped ≥ 0; `w` unused |
| *(later)* `sh` | `float` ×48 (or `half`) | 192 (96) | 16 coefficients × RGB, coefficient-major |

**Why `cov3d` is split 4 + 2:** six floats is 24 bytes, which isn't a multiple of 16, so one `float6` array can't use 128-bit vector loads. The split gives one `LDG.128` plus one `LDG.64` instead of six scalar loads. Small, but it's the same reasoning as your `float4` means, applied consistently.

**Where Σ is computed:** upload `scale` (`float4`, padded) and `rotation` (`float4` in **wxyz**, stated explicitly) into *temporary* device buffers. Run a one-shot `build_cov3d` kernel that uses the shared `SPLAT_HD` function. Write `cov3d_a/b`, then let the temporaries go out of scope (RAII frees them). The CPU oracle calls the *same* function on the host. This removes the Eigen `Matrix3f` per Gaussian in `Scene` (36 B × N of host memory) and the review C6 hazard.

### Per-Gaussian, per-frame buffers (written by preprocess)

| Buffer | Type | Bytes | Read by |
|---|---|---|---|
| `mean2d` | `float2` | 8 | duplicate (rect), render |
| `depth` | `float` | 4 | duplicate (key) |
| `radius` | `int` | 4 | duplicate (rect); 0 ⇔ culled |
| `conic_opacity` | `float4` | 16 | render: `(A, B, C, opacity)` of Σ₂D⁻¹ |
| `rgb` | `float4` | 16 | render (needed once SH is view-dependent; with SH0 the render kernel can read the scene `rgb` directly) |
| `tiles_touched` | `uint32_t` | 4 | scan input |
| `offsets` | `uint32_t` | 4 | scan output (inclusive); duplicate |

### Per-instance buffers

| Buffer | Type | Bytes |
|---|---|---|
| `keys[2]` (DoubleBuffer) | `uint64_t` | 16 |
| `vals[2]` (DoubleBuffer) | `uint32_t` | 8 |

### Per-image buffers

| Buffer | Type | Size |
|---|---|---|
| `tile_ranges` | `uint2` (start, end) | 8 B × tiles |
| `image` (Phase A only; Phase B writes into the mapped PBO) | `uchar4` | 4 B × pixels |
| *(debug)* `n_contrib` | `uint32_t` | 4 B × pixels: how many Gaussians were blended before termination |

### SoA vs AoS, decided per kernel

Ask how each kernel's threads access the data:

- **Preprocess:** thread *i* reads Gaussian *i*. Consecutive threads read consecutive elements, so **SoA** is fully coalesced. A warp loading `mean_opacity` touches 32 × 16 B = 512 B = exactly four 128-byte lines.
- **Render:** each thread loads Gaussian `point_list[k]`, a **gather** by sorted id, which is effectively random. Every array read during that gather touches a different cache sector. Reading `mean2d` + `conic_opacity` + `rgb` from three SoA arrays costs ~3 sectors per Gaussian. A 32-byte packed `RenderPacket {float2 xy; float4 conic_o; half4 rgb}` written by preprocess would cost ~1.
- **Recommendation:** start with SoA (it's simpler, and it's what the reference does). Later, measure *sectors/request* on the render kernel's gather in Nsight Compute, try the packed AoS, and compare. Running that comparison yourself teaches more about memory systems than any amount of reading.

**A coalescing nuance an interviewer may probe:** a `float3` array is still *contiguous*, so a warp touches the same cache lines as a tightly packed array would. Coalescing isn't the problem. The cost is that 12-byte elements aren't 16-byte aligned, so the compiler issues three 32-bit loads per thread instead of one 128-bit load. More instructions, same bytes.

---

## 4. Memory budget

Per-Gaussian bytes (SH0): scene 56 + per-frame 56 = **112 B**. Per instance: **24 B** (+ small sort temp with `DoubleBuffer`). Per pixel: 4 B. Per tile: 8 B.

$$
\text{Bytes} \approx 112\,N \;+\; 24 \cdot 1.5\,L_{\text{peak}} \;+\; 4\,WH \;+\; \text{CUB temp}
$$

The factor 1.5 is the grow-only headroom.

| Scenario | Scene + per-Gaussian | Binning (L) | Image | Total |
|---|---|---|---|---|
| N = 1M, 1080p, L/N ≈ 8 (L = 8M) | 112 MB | 192 MB (288 MB with headroom) | 8 MB | **≈ 0.3–0.4 GB** |
| N = 3M, 1080p, L/N ≈ 8 (L = 24M) | 336 MB | 576 MB (864 MB) | 8 MB | **≈ 0.9–1.2 GB** |
| N = 3M with SH3 (+176 B/Gaussian) | 864 MB | 576 MB (864 MB) | 8 MB | **≈ 1.5–1.8 GB** |

**What to take away:**

- **L, not N, is the variable to watch.** L depends on the view. Zooming in makes every ellipse cover more tiles, so L can grow by an order of magnitude while N stays fixed. That is why binning is grow-only and why `FrameStats` reports L every frame. Put it on screen.
- **SH3 dominates the scene memory.** That is the usual reason viewers store SH in `half`.
- **A 16×16 tile is a trade-off.** Smaller tiles mean more instances per Gaussian (larger L, more sort work) but less wasted per-pixel evaluation. Larger tiles mean the reverse. 16×16 = 256 threads also happens to be a good block size.

---

## 5. Streams and synchronisation

### Use one explicit non-blocking stream

`cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`, owned by `Impl` through RAII. Every API call gets the stream: `cudaMemsetAsync`, `cudaMemcpyAsync`, CUB's `stream` parameter, and `<<<grid, block, smem, stream>>>`.

**Why not the default stream:** the *legacy* default stream implicitly synchronises with every other blocking stream. Any library or code path that touches it (a stray `cudaMemset`, for example) silently serialises the whole pipeline. An explicit stream makes the ordering what your code says it is.

### The complete list of host↔device sync points

| # | Where | Why it exists | Can it be removed? |
|---|---|---|---|
| 1 | After the scan, to read **L** | CUB's radix sort takes `num_items` as a **host** integer, and the binning capacity must be ensured on the host | Not without an upper-bound allocation (wasteful) or device-side launch sizing. **Accept it**; it's ~10 µs of latency |
| 2 | Phase A: after the image D2H copy | The host needs the pixels for `glTexSubImage2D` | **Yes, with Phase B interop** |
| 3 | `glfwSwapBuffers` with vsync | Blocks to the display refresh rate | Call `glfwSwapInterval(0)` when *measuring*; otherwise FPS is capped at the refresh rate and tells you nothing |

For sync point 1, copy `offsets[N-1]` into a **pinned** `uint32_t` with `cudaMemcpyAsync`, then `cudaStreamSynchronize(stream)` (or `cudaEventSynchronize` on an event recorded after the copy). Don't use `cudaDeviceSynchronize`, which waits for *everything* on the device, not just your stream.

---

## 6. Every transfer, enumerated

| Transfer | Direction | Size | Frequency | Memory | Notes |
|---|---|---|---|---|---|
| PLY → `HostScene` | disk → host | file size | once | pageable | Current loader reads the whole file into memory and then parses it, so peak memory is ~2× the file. Fine for learning |
| `HostScene` → `SceneBuffers` | H2D | ~56 B × N | once per scene | pageable is fine | Pinning a one-off buffer costs about as much as the copy saves |
| `CameraParams` | kernel argument | ~100 B | per frame | constant bank | Replaces today's `cudaMemcpy` of the view matrix |
| `L` | D2H | 4 B | per frame | **pinned** | Sync point 1 |
| Image (Phase A) | D2H, then H2D via `glTexSubImage2D` | 8.3 MB at 1080p, **twice** | per frame | **pinned** host staging | ~0.4–0.7 ms per crossing on PCIe 3/4 ×16 in practice, plus the sync. Visible in Nsight Systems |
| Image (Phase B) | device-only (PBO → texture) | 8.3 MB | per frame | VRAM | No PCIe traffic |

### Pinned vs pageable, from first principles

The GPU's copy engine is a DMA device. It reads *physical* memory directly and needs those pages to stay resident and in place for the whole transfer. Pageable memory can be swapped out or moved by the OS at any time. So for a pageable copy, the driver first `memcpy`s into its own internal pinned bounce buffer and DMAs from there. That's one extra CPU copy, and the call becomes synchronous. `cudaMallocHost` gives you page-locked memory the DMA engine can use directly, which enables truly asynchronous copies.

**Failure mode of over-pinning:** pinned pages can't be swapped, so pinning gigabytes starves the OS. Pin only small, hot, per-frame staging buffers.

---

## 7. CUDA–OpenGL interop (Phase B)

**Goal:** the render kernel writes pixels into memory that OpenGL can draw, and the image never crosses PCIe.

**Mechanism (PBO path, recommended first):**

1. **Once, after creating the PBO:** `glBufferData(GL_PIXEL_UNPACK_BUFFER, W*H*4, nullptr, GL_STREAM_DRAW)`, then `cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsRegisterFlagsWriteDiscard)`. *WriteDiscard* tells the driver that CUDA overwrites everything, so the old contents need not be preserved.
2. **Each frame:** `cudaGraphicsMapResources(1, &res, stream)` → `cudaGraphicsResourceGetMappedPointer(&ptr, &size, res)` → the render kernel writes `uchar4` to `ptr` → `cudaGraphicsUnmapResources(1, &res, stream)`.
3. **GL side:** bind the PBO to `GL_PIXEL_UNPACK_BUFFER` and call `glTexSubImage2D(..., (void*)0)`. With a PBO bound, the last argument is an *offset* into the PBO, not a host pointer, so the copy is GPU-to-GPU.
4. **On resize:** unregister, then resize the PBO and texture, then re-register. On shutdown, unregister *before* deleting the PBO and before the context dies (the order in [02 §4](02-architecture.md#4-construction-and-destruction-order)).

**Alternative (texture path):** `cudaGraphicsGLRegisterImage` on the texture, map it, get a `cudaArray_t`, create a `cudaSurfaceObject_t`, and write with `surf2Dwrite`. This skips the PBO→texture copy, but the kernel writes through the surface API (byte-addressed x coordinate, a gotcha) instead of plain pointers. Do the PBO version first.

**Device-selection gotcha:** interop only works if CUDA runs on the same GPU that owns the GL context. Before any other CUDA call, use `cudaGLGetDevices(&count, devs, 1, cudaGLDeviceListAll)` and `cudaSetDevice(devs[0])`. On hybrid-graphics laptops (GL on the integrated GPU), registration fails or silently routes through host memory. That's one of the open questions in the [README](README.md#decisions-this-plan-assumes-and-open-questions). Keep Phase A as the fallback, selected at runtime.
