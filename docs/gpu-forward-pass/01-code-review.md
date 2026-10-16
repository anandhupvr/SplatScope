# 01 — Code Review of `gpu_renderer` @ `12208c2`

**Mental model for reading this review:** a GPU pipeline's correctness rests on *invariants between stages*: what one kernel writes, the next one assumes. Most bugs below are broken invariants rather than bad arithmetic. For each finding I state the invariant it breaks, so the fix generalises.

Severity legend:

| Tag | Meaning |
|---|---|
| 🔴 **Correctness** | Wrong pixels, undefined behaviour, out-of-bounds writes, or a crash |
| 🟠 **Resource / ownership** | Leaks, double frees, lifetime hazards, unchecked errors |
| 🟡 **Performance** | Correct but needlessly slow, in ways that will mislead your measurements |
| 🔵 **Design / hygiene** | Makes the next feature harder or the code misleading |

## What's already good (keep these instincts)

- **Separating `SceneBuffers` from `ScratchBuffers`** is the right first cut. Data that lives as long as the scene and data that is per-frame have different allocation policies. [03](03-memory-and-data-movement.md) formalises this into lifetime classes.
- **Padding means to `float4`** for 16-byte coalesced loads, along with your comment working out the transaction maths (`gaussian_pipeline.cu:154`). That is the right reasoning.
- **Packing `conic` + `opacity` into one `float4`** is exactly what the reference rasterizer does, because the render kernel needs both together.
- **Using CUB for the scan** instead of hand-rolling one, and querying temp storage size up front (`gaussian_pipeline.cu:287-290`).
- **Decoding activations at load time** (`exp` for scale, `sigmoid` for opacity, SH-DC→RGB) and filtering non-finite Gaussians (`ply_loader.cpp:85-133`). Bad data stops at the boundary.
- **The `Renderer` interface** with CPU and GPU implementations. It is the right seam, though [02](02-architecture.md) moves presentation out of it.
- **The checkpoint discipline:** writing white dots at projected centres (`gaussian_pipeline.cu:143`) before building the rest is exactly how to bring up a GPU pipeline.

---

## A. GPU pipeline correctness

### A1 🔴 Inverted guard in `duplicate_and_generate_keys`

`gaussian_pipeline.cu:34`

```cpp
if (d_radii[idx] > 0) return;   // returns for every VISIBLE Gaussian
```

**Invariant broken:** "a Gaussian emits exactly `tiles_touched[idx]` instances, starting at `offsets[idx-1]`". Visible Gaussians emit nothing. Culled ones fall through and read stale radii (see A2), then write keys.

**Fix:** `if (radii[idx] <= 0) return;` A cleaner equivalent is `if (tiles_touched[idx] == 0) return;`, because that is the quantity the scan used.

### A2 🔴 Per-frame outputs not written for culled Gaussians

`gaussian_pipeline.cu:89-97`. The kernel `return`s before writing `d_radii`, `d_tiles_touched`, `d_depths` and `d_means2d`.

**Invariant broken:** "every slot of every per-Gaussian output is written every frame". `cudaMalloc` does not zero memory. On frame 1 these slots hold garbage. On later frames they hold the values from whatever frame last saw that Gaussian as visible. The inclusive scan then sums garbage, so `L` is garbage. That is why `cudaMalloc(L * 8)` can request absurd sizes, and why the `printf` count jumps around between frames.

**Fix:** at the top of the preprocess kernel, after the bounds check, write `radii[idx] = 0; tiles_touched[idx] = 0;` before any early-out. Do not "fix" this with a `cudaMemset` per frame: that costs an extra pass over memory and hides the real rule.

**Tooling lesson:** `compute-sanitizer --tool initcheck ./SplatScope scene.ply` reports exactly this class of bug, a read of uninitialised device memory. Run it once per milestone.

### A3 🔴 Depth packed by value-cast instead of by bit pattern

`gaussian_pipeline.cu:55-56`

```cpp
key |= (uint32_t)depth;   // 3.7f → 3, 3.2f → 3: tie
```

**Why it matters:** the sort key is `[tile:32 | depth:32]`. A value cast truncates, so every Gaussian in a tile within the same integer depth unit has the same key. Their relative order is then whatever the radix sort's stability preserves from emission order, which is Gaussian-index order and unrelated to depth. You get wrong compositing order and view-dependent flicker.

**Fix:** `key |= __float_as_uint(depth);`. For *positive* IEEE-754 floats, the raw bit pattern compared as an unsigned integer sorts in the same order as the float value. The sign bit is 0, the exponent comes next in the high bits, then the mantissa. The near-plane cull guarantees depth > 0. See [04 §3](04-cuda-forward-pipeline.md#3-duplicate-with-keys).

The comment `// gaussian id` on that line is also wrong. The id goes in the *value*, not the key.

### A4 🔴 Tile rectangle computed two different ways → out-of-bounds writes

Count pass (`gaussian_pipeline.cu:128-136`):

```cpp
tile_min_x = max(0, floorf((u - radius) / 16.0f));  tile_max_x = min(tW, ceilf((u + radius) / 16.0f));
```

Write pass (`gaussian_pipeline.cu:42-45`):

```cpp
x_min = floorf((point.x - radius) / tile_grid.x);  // tile_grid.x = NUMBER of tiles (e.g. 50), not tile SIZE
x_max = ceilf((point.x + radius) / tile_grid.x);   // and no clamping
```

**Invariant broken:** *the count pass and the write pass must compute bit-identical rectangles.* The scan gives each Gaussian exactly `tiles_touched[i]` slots. If the write pass iterates over a bigger rectangle, it overwrites the next Gaussian's slots. At the last Gaussian, it writes past the end of the allocation. The `printf("error: negative tile index")` at line 51 is this bug showing itself.

**Fix:** write **one** `SPLAT_HD` function, `get_tile_rect(mean2d, radius, tile_grid) -> {min, max}` with clamped, exclusive-max semantics, and call it from both kernels (and from the CPU oracle). This is the single most important structural fix in the pipeline. See [04 §1.6](04-cuda-forward-pipeline.md#16-screen-rectangle-in-tile-units).

### A5 🔴 Covariance projection is transposed: $J^\top\Sigma'J$ instead of $J\Sigma'J^\top$

`gaussian_math.cuh:35-44, 76`. `glm::mat3(...)` takes its arguments **column by column**. The code passes column 0 = `(fx/z, 0, 0)`, column 1 = `(0, fy/z, 0)` and column 2 = `(-fx·x/z², -fy·y/z², 0)`. The resulting matrix *is* the padded Jacobian $J$ (row 0 = `[fx/z, 0, -fx·x/z²]`). The code then computes `transpose(J) * sigma_cam * J`, which is $J^\top\Sigma'J$.

I checked this numerically with $f=800$, $\mathbf t=(0.5, 0.2, 3)$ and a random SPD $\Sigma'$. The 2×2 blocks of $J^\top\Sigma'J$ and $J\Sigma'J^\top$ differ (≈58.8k vs ≈42.5k in the (0,0) entry). They only agree when $x = y = 0$, i.e. for a Gaussian on the optical axis. On screen you see ellipses that are right in the centre and increasingly wrong toward the edges.

**Why the reference code *looks* like yours:** Kerbl et al. write the constructor arguments in *row* order. That means their `glm::mat3 J` is really $J^\top$, and their `W` is really $R^\top$. They then compute `transpose(T) * Σ * T` with `T = W·J`, which works out to $(JR)\,\Sigma\,(JR)^\top$. Copying half of that convention flips the result.

**Fix:** compute $\Sigma_{2D} = J\,(R\,\Sigma\,R^\top)\,J^\top$, keeping only the top 2×2. Better, use the shared math header ([05](05-cpp-implementation-guide.md#4-shared-math-for-host-and-device)), which has an explicit 2×3 `J`. That removes the padded-3×3 confusion entirely. Then unit-test it against the CPU `splat::project_covarience`, which *is* correct (`splat.cpp:51`).

**Naming:** the function is called `compute_cov3d_inv`, but it returns the **2D covariance**, not an inverse of anything. Rename it to `project_cov2d`.

### A6 🔴 Low-pass dilation multiplied; eigenvalues computed from the wrong matrix

`gaussian_pipeline.cu:110-125`

```cpp
float a = cov2d.x * 0.3f;    // should be +
float c = cov2d.z + 0.3f;
...
float mid = 0.5f * (cov2d.x + cov2d.z);   // uses UN-dilated a, c
float disc = sqrtf(fmaxf(0.0f, mid * mid - det));  // det is from the DILATED matrix
```

**What the +0.3 is:** convolution with a pixel-sized isotropic Gaussian of variance 0.3 px². The covariance of a convolution of Gaussians is the *sum* of the covariances. That is why it is `+0.3·I`. It guarantees every splat covers at least about a pixel (anti-aliasing) and keeps the matrix well-conditioned for inversion. Multiplying by 0.3 shrinks every splat's x-variance by 70%.

**Fix:** dilate first, then derive `det`, the conic and the eigenvalues all from the *same* dilated `(a, b, c)`. If `det <= 0`, cull: set radius 0 and return.

### A7 🔴 Centre-on-screen culling and integer pixel centres

`gaussian_pipeline.cu:93-100`

1. `if (u < 0 || u >= W || ...) return;` culls any Gaussian whose **centre** is off-screen, even when its 3σ footprint covers half the screen. Large splats pop in and out at the image border as you orbit. **Fix:** cull only when the clamped tile rectangle has zero area.
2. `int u = (int)(...)` truncates the projected mean. The render kernel evaluates $\exp(-\tfrac12 d^\top\Sigma^{-1}d)$ relative to the mean, so quantising the mean to whole pixels makes small splats "swim" in one-pixel steps during camera motion. **Fix:** keep `mean2d` as `float2`.

### A8 🔴 No Jacobian guard near the frustum edge

The Jacobian's off-diagonal terms are $-f\,x/z^2$. For a Gaussian just outside the view frustum and close to the camera, $x/z$ is large and the linearisation of the projection is poor. The projected covariance blows up into screen-filling smears. The reference implementation clamps $x/z$ and $y/z$ to $\pm 1.3\tan(\text{fov}/2)$ *before* building $J$ (only for $J$, not for the mean). Add this. It is one line, and its absence produces some of the most confusing artifacts you'll see when zooming in. Both the CPU (`splat.cpp:34-42`) and GPU paths lack it.

### A9 🟡 Double-precision literal in device code

`gaussian_math.cuh:31`: `float inv_z = 1.0 / cam.z;`. `1.0` is a `double`, so this does an FP64 divide. Consumer GPUs run FP64 at 1/32–1/64 of FP32 throughput. Use `1.0f`. **Habit:** compile with `-Xptxas -warn-double-usage` (or grep the SASS for `DADD`/`DMUL`) to catch these.

### A10 🔵 Homogeneous divide on a rigid transform

`gaussian_math.cuh:14-21` and `splat.cpp:18-24` divide by `w + 1e-7`. The view matrix is rigid ($[R\,|\,t]$, bottom row $[0,0,0,1]$), so $w \equiv 1$. The divide is harmless but costs a division per Gaussian, and it suggests "projective 4×4", which is the wrong mental model here. In 3DGS, projection happens through the intrinsics and the Jacobian, not a clip-space matrix. Pass $R$ (3×3) and $t$ (3) instead of a 4×4.

---

## B. Camera and coordinate conventions

### B1 🔴 The view matrix is a reflection (det = −1), cancelled by OpenGL's texture rows

`camera.cpp:121-133`

```cpp
right = forward.cross(Vector3f(0,-1,0));   // "to match COLMAP"
up    = right.cross(forward);
view rows = [right; up; forward]
```

Worked example at `yaw = pitch = 0` (eye at `(0,0,5)`, looking at the origin): `forward = (0,0,-1)`, `right = (-1,0,0)`, `up = (0,-1,0)`. Stacking those rows gives $R = -I$, so $\det R = -1$. That is a point reflection, not a rotation. In general, rows `(r, r×f, f)` form a left-handed basis.

Why the image still looks right: camera $+y$ here is world "up-ish". The projection sends $+y$ to larger $v$, i.e. later rows in memory. `glTexImage2D` treats row 0 as the *bottom* of the texture, so the two vertical flips cancel.

**Why fix something that "works":** two cancelling sign errors become one visible error as soon as either side changes. Each of these breaks it: switching to CUDA-GL interop with a different row order, saving a PNG (row 0 = top), loading a COLMAP camera pose, or adding SH evaluation that uses camera-frame directions. Also, $R\Sigma R^\top$ happens to be invariant to $R \to -R$, so the covariance won't warn you.

**Fix:** define the conventions once ([02 §6](02-architecture.md#6-coordinate-conventions-single-source-of-truth)). Use camera frame = OpenCV (x right, **y down**, z forward), $\det R = +1$, image row 0 = top. Do the *one* flip at presentation time by flipping the quad's texture v-coordinate. Add a unit test: `assert(abs(R.determinant() - 1) < 1e-5)`.

### B2 🔵 Pixel-centre convention is implicit

The CPU loop evaluates the Gaussian at integer `(x, y)`, so pixel centres sit at integer coordinates (the OpenCV convention). The intrinsics `cx = W/2` assume the same. That's fine, but write it down. If anyone later evaluates at `x + 0.5`, every splat shifts by half a pixel and CPU-vs-GPU tests fail by tiny, confusing amounts.

---

## C. Resource management and ownership

### C1 🟠 Leaks in the free functions

- `free_scene` (`gaussian_pipeline.cu:327-339`) never frees `d_cov3d`.
- `free_scratch` (`:307-325`) never frees `d_cub_temp`.
- `set_scene` called twice leaks everything (the comment at `gpu_renderer.cpp:45` already admits this).

These aren't three bugs. They are one design gap: **resources whose release is a separate manual call**. Every new buffer needs edits in three places (struct, alloc, free), and forgetting one is silent. RAII makes the leak structurally impossible ([05 §1](05-cpp-implementation-guide.md#1-raii-wrappers-for-cuda-resources)).

### C2 🟡🟠 Per-frame `cudaMalloc` / `cudaFree` of the binning buffers

`gaussian_pipeline.cu:233-236, 269-272`. `cudaFree` implicitly synchronises the whole device, and `cudaMalloc` is a heavyweight driver call. Doing four of each per frame puts unbounded, jittery stalls in the frame and makes any timing you take meaningless. **Policy:** grow-only capacity ([03 §2](03-memory-and-data-movement.md#2-allocation-policy-per-class)).

### C3 🟠 Implicitly copyable owners → double free

`GpuRenderer`, `Display`, `Window` and `Shader` all own raw handles (`void*` device pointers, GL names, `GLFWwindow*`). None of them deletes the copy constructor. Any accidental copy, such as a by-value parameter or a `std::vector` reallocation, double-frees. `Viewer`'s initialisers `window_(Window(...))` and `display_(Display(w,h))` only work because C++17 guarantees copy elision for prvalues. One refactor to a non-prvalue and it breaks.

**`Window` is worse:** it registers `this` with `glfwSetWindowUserPointer` (`window.cpp:22, 34`). A *move* would leave GLFW pointing at the moved-from object. **Rule:** `Window` is non-copyable **and** non-movable. Hold it by value in its single owner, or in a `unique_ptr` if it must be relocated.

### C4 🟠 Unchecked CUDA calls and async errors

- Kernel launches are never checked. Launch-configuration errors (e.g. `grid = 0` when N = 0) come back from `cudaGetLastError()` and are otherwise lost.
- `cudaDeviceSynchronize()` (`:214`), the L readback `cudaMemcpy` (`:227`), the frame `cudaMemcpy` (`:263`) and every `cudaFree` are unchecked.
- **Gotcha an interviewer will probe:** kernel *execution* errors are **asynchronous and sticky**. An illegal address in the render kernel is reported by whatever CUDA call happens to synchronise next, often a `cudaMemcpy` several lines later. After that the context is unusable; you can't recover in-process. Hence the policy in [02 §7](02-architecture.md#7-error-handling-policy): check every launch cheaply, and in Debug builds synchronise after each kernel so the error points at the kernel that caused it.

### C5 🟠 Implicit lifetime contract: `const Scene* scene_`

`gpu_renderer.h:23` and `cpu_renderer.h:17`. The renderer stores a raw pointer to a `Scene` owned by `Viewer`. It is correct today only because `Viewer` declares `scene_` before `renderer_`. Nothing enforces that, and the GPU renderer doesn't even need the host scene after upload. See [02 §3](02-architecture.md#3-ownership-graph) for the explicit policy: the GPU renderer **copies on upload and retains nothing**; the CPU renderer holds a `shared_ptr<const HostScene>`.

### C6 🔴 `reinterpret_cast` of Eigen containers to `float*`

`gpu_renderer.cpp:48-50`:

- `Eigen::Quaternionf` stores its coefficients in memory as **(x, y, z, w)**, while the PLY order is `rot_0..3 = (w, x, y, z)`. The loader's `Quaternionf(w,x,y,z)` constructor (`ply_loader.cpp:74`) is right. But the bytes you upload are **xyzw**, and any kernel reading them as wxyz computes a wrong rotation. It's latent today (the GPU uses CPU-computed `cov3d`), but it becomes live the day you move covariance construction to the GPU.
- `std::vector<Eigen::Vector3f>` happens to be tightly packed (12-byte stride, no padding), but that is an Eigen layout detail you're silently depending on.

**Fix:** have the loader produce plain SoA arrays already in the GPU layout ([03 §3](03-memory-and-data-movement.md#3-layouts)). The upload then becomes a straight `memcpy` of `std::vector<float4>`, with no casts and no hidden layout assumptions.

### C7 🔵 Uninitialised members

`ScratchBuffers::width, height` have no initialisers (`gaussian_pipeline.cuh:26`). `L_max` is declared but never used. `GpuRenderer::scene_` is uninitialised until `set_scene`, so `render()` before `set_scene()` is UB.

---

## D. Performance (things that will distort your measurements)

### D1 🟡 `Scene::covarience_3d()` returns a full copy

`scene.h:12`, `scene.cpp:13-15`: it returns `std::vector<Eigen::Matrix3f>` **by value**. `CpuRenderer::render` calls it **every frame** (`cpu_renderer.cpp:51`). Binding the result to `const auto&` extends the temporary's lifetime, but it doesn't avoid the copy. For 3M Gaussians that's 108 MB of copying per frame. **Fix:** return `const std::vector<...>&`. (Better: the precomputed covariance belongs to the renderer that needs it; see [03](03-memory-and-data-movement.md).)

### D2 🟡 Heap allocation per pixel per splat in the CPU renderer

`framebuffer.cpp:29-36`: `get_pixel` returns `std::vector<int>`, which means one `malloc` and one `free` per pixel touched per Gaussian. That is easily the CPU renderer's dominant cost. **Fix:** blend into a `std::vector<float>` RGB accumulation buffer in place. See also E3.

### D3 🟡 Texture re-specified and mipmapped every frame

`display.cpp:60-69`: `glTexImage2D` re-allocates texture storage every frame. `glGenerateMipmap` builds a full mip chain every frame that is never sampled (`MIN_FILTER = GL_LINEAR`, not a mipmap filter). **Fix:** call `glTexStorage2D` once (immutable storage) and `glTexSubImage2D` per frame, with no mipmaps. Also switch the wrap mode to `GL_CLAMP_TO_EDGE`: with `GL_REPEAT` and linear filtering, the top border samples bleed from the bottom row.

### D4 🟡 Default stream, device-wide sync, pageable copies

The whole pipeline runs on the legacy default stream, with `cudaDeviceSynchronize` and pageable `cudaMemcpy`. Each pageable D2H copy is synchronous *and* staged through a driver bounce buffer. It works, but it hides where the real sync points are. See [03 §5](03-memory-and-data-movement.md#5-streams-and-synchronisation).

### D5 🟡 Uploading the view matrix with `cudaMemcpy` every frame

`gaussian_pipeline.cu:192`. A 64-byte matrix doesn't need a device buffer. Pass a `CameraParams` struct **by value** as a kernel argument. Kernel parameters live in the constant bank and are broadcast to every thread at no extra cost (limit 4 KB; 32 KB on Volta+ with CUDA ≥ 12.1).

### D6 🟡 `printf` inside kernels in the hot path

`gaussian_pipeline.cu:51, 231`. A device-side `printf` serialises through a ring buffer. Keep it behind a debug macro.

---

## E. The CPU renderer (it is your test oracle, so it must be right)

The CPU path is valuable *only* if it implements the **same model** as the GPU path. Today it differs in several places, so a CPU-vs-GPU diff would flag "differences" that are really oracle bugs.

| # | Issue | Where | Fix |
|---|---|---|---|
| E1 🔴 | The bbox half-extent is $\sqrt{3\sigma^2}=1.73\sigma$, but the cutoff is $d^2>9$ (3σ). Splats are clipped to a 1.73σ box, so you see square-ish edges | `splat.cpp:62-63` | half-extent = $3\sqrt{\lambda_{max}}$ (same radius as the GPU) |
| E2 🔴 | Row loop `y < box.w()` but column loop `x <= box.z()`: the last row is skipped | `cpu_renderer.cpp:116-117` | Use the shared `get_rect` semantics (exclusive max) for both |
| E3 🔴 | Back-to-front blending done **in uint8 with truncation**. Each over-operation rounds *down*, so many layers drift darker and band | `cpu_renderer.cpp:127-132` | Accumulate in float; convert once at the end with round-to-nearest |
| E4 🔴 | Regularisation `+1e-4` vs the GPU's `+0.3`; near plane `0` vs `0.2`; no α ≥ 1/255 skip; no T < 1e-4 termination | various | One `RenderSettings` struct consumed by both backends |
| E5 🔴 | `FrameBuffer` was resized to RGBA (`×4`) but `clear`/`set_pixel`/`get_pixel` still stride by 3, so the CPU path is broken on this branch (acknowledged in your comment) | `framebuffer.cpp:6-36` | One pixel format (RGBA8) for presentation; the CPU renderer accumulates in float RGB internally |

---

## F. Build, dependencies, hygiene

| # | Issue | Where | Recommendation |
|---|---|---|---|
| F1 🔴 | **A fresh clone doesn't build.** `external/glm` and `external/tinply` are not in the repo (tinply is `.gitignore`d), and the loader includes `example-utils.hpp` from tinyply's *examples* | `CMakeLists.txt:24,58`, `ply_loader.cpp:10` | `FetchContent` with pinned tags; replace `read_file_binary`/`memory_stream` with a 20-line `std::ifstream` read |
| F2 🔵 | `project(... LANGUAGES CUDA C CXX)` makes CUDA mandatory to even *configure*; no CPU-only build | `CMakeLists.txt:3` | `option(SPLATSCOPE_WITH_CUDA ON)` + `enable_language(CUDA)` inside it; the CUDA backend becomes a separate static library |
| F3 🔵 | `CUDA_SEPARABLE_COMPILATION ON` (relocatable device code) is unnecessary: there are no cross-TU `__device__` calls. It adds a device-link step and blocks cross-TU inlining | `CMakeLists.txt:75-77` | Remove it until you actually need it |
| F4 🔵 | `GLM_FORCE_*` defined for *every* TU; three math representations (Eigen on host, glm on device, raw `float*` in between) | `CMakeLists.txt:48-54` | One shared `SPLAT_HD` math header ([05 §4](05-cpp-implementation-guide.md#4-shared-math-for-host-and-device)) |
| F5 🔵 | `Shader.h` has no `#pragma once`; `setVec4` ignores `name` and hard-codes `"fColor"`; the program object is never deleted; the error path continues with empty source | `Shader.h`, `Shader.cpp:32-34,88` | RAII `GlProgram`; throw on compile/link failure |
| F6 🔵 | `glfwInit`/`glfwCreateWindow` results unchecked; `on_key_` and `on_resize_` are never registered, so `key_shift`, `key_ctrl` and `fb_width` are dead fields; the resize lambda changes the viewport but not the render resolution; on HiDPI screens the window size ≠ framebuffer size | `window.cpp` | See [02 §5](02-architecture.md#5-per-frame-control-flow) on resize |
| F7 🔵 | `main` doesn't catch exceptions, so a bad path calls `std::terminate` with no message | `main.cpp:14-19` | Catch at the composition root, print, return 1 |
| F8 🔵 | Spelling that leaks into the API: `Instrinsic`, `covarience`, `set_intrinic`, `view_roataion`, `poisitions` | various | Fix now, while there are few call sites |
| F9 🔵 | `splat::compute_covariance(const Vector3f&, Quaternionf&)` takes the rotation by **non-const** reference | `splat.h:9` | `const Eigen::Quaternionf&` |

---

## Severity-ordered fix list

1. A2 → A1 → A4 → A3 (the binning chain; until all four are fixed, nothing downstream can be trusted)
2. A5, A6, A8 (projection maths), checked against the CPU oracle after E1–E5
3. A7 (float means, rect-based culling)
4. C1–C4 by introducing RAII wrappers (this removes the bug class, not just the instances)
5. B1 conventions, before you add interop or PNG dumps
6. C2/D-series once there is a stage timer to measure the improvement
7. F-series alongside the CMake restructure
