# 04 — The CUDA Forward Pipeline, Kernel by Kernel

> **Mental model:** *sorting is the hard part of transparency, and the tile trick turns one impossible per-pixel sort into a single global radix sort.* Each pixel needs its Gaussians in depth order. Sorting per pixel is far too much work. Instead, every (Gaussian, tile) pair gets a 64-bit key `[tile | depth]`, and **one** radix sort of all those keys leaves each tile's Gaussians contiguous *and* depth-ordered. After that, every tile is an independent, embarrassingly parallel compositing problem.

Notation: N = Gaussians, L = instances (Gaussian×tile pairs), T = tiles, `BLOCK = 16`.

Each stage below gives: **what it computes → the maths → the invariant it must uphold → the pseudocode shape → common failure modes.** Kernels are shown as pseudocode; signatures are in [05](05-cpp-implementation-guide.md).

---

## 0. One-time: build Σ (3D covariance) from scale and rotation

**Why parameterise as scale + rotation instead of storing Σ directly?** Because a covariance must be symmetric positive semi-definite, and an optimiser updating six raw numbers would break that. Writing $\Sigma = R S S^\top R^\top = (RS)(RS)^\top$ makes it PSD by construction. For a viewer this is inherited: the PLY stores what training optimised.

**Geometric picture:** start with a unit sphere, stretch it along the x/y/z axes by $s_x, s_y, s_z$ (the matrix $S$), then rotate it ($R$). The result is an oriented ellipsoid with semi-axes of length $s_i$ along the columns of $R$.

With a normalised quaternion $(w,x,y,z)$:

$$
R=\begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix},\qquad
M = R\,\mathrm{diag}(s),\quad \Sigma = M M^\top
$$

$M = R\,\mathrm{diag}(s)$ just scales *column* $i$ of $R$ by $s_i$, so there's no need for a full matrix product. Store the six unique entries (split 4 + 2, see [03 §3](03-memory-and-data-movement.md#3-layouts)).

**Failure mode:** an unnormalised quaternion makes $R$ non-orthogonal, which silently scales the ellipsoid. The loader normalises (good). Keep the quaternion component order **explicit in the type name** (`quat_wxyz`), because Eigen stores xyzw (review C6).

---

## 1. Preprocess (one thread per Gaussian)

### 1.1 Initialise outputs (the A2 fix)

```
if idx >= N: return
radius[idx] = 0; tiles_touched[idx] = 0          // every slot written, every frame
```

**Invariant:** `radius[idx] == 0  ⇔  tiles_touched[idx] == 0  ⇔  Gaussian idx is culled this frame`.

### 1.2 World → camera, near cull

$$\mathbf t = R_{cw}\,\boldsymbol\mu + \mathbf t_{cw}\qquad(\text{rigid; no } w \text{ divide})$$

Cull if $t_z < z_{near}$ ($0.2$ in the reference). The near cull isn't just about "behind the camera". The Jacobian has $1/z$ and $1/z^2$ terms, so Gaussians very close to the camera have exploding projected covariances. The near plane keeps the linearisation sane.

### 1.3 Project the mean

$$u = f_x\,\frac{t_x}{t_z} + c_x,\qquad v = f_y\,\frac{t_y}{t_z} + c_y$$

Keep $(u,v)$ as **floats** (review A7). Pixel centres sit at integer coordinates ([02 §6](02-architecture.md#6-coordinate-conventions-single-source-of-truth)).

### 1.4 Project the covariance (EWA splatting)

**Problem:** perspective projection is non-linear, and a Gaussian pushed through a non-linear map is no longer Gaussian.

**Solution:** linearise the projection around the mean. A Gaussian through a *linear* map $A$ stays Gaussian with covariance $A\Sigma A^\top$. So use the first-order Taylor expansion $\pi(\mathbf t + \delta) \approx \pi(\mathbf t) + J\delta$:

$$
J = \frac{\partial(u,v)}{\partial(x,y,z)}\bigg|_{\mathbf t}
= \begin{bmatrix} f_x/t_z & 0 & -f_x t_x / t_z^2 \\ 0 & f_y/t_z & -f_y t_y/t_z^2 \end{bmatrix}
\quad (2\times 3)
$$

$$
\boxed{\Sigma_{2D} = J\,R_{cw}\,\Sigma\,R_{cw}^\top\,J^\top}\qquad(2\times 2)
$$

Read right to left: rotate the ellipsoid into the camera frame, then apply the local linear projection.

**Frustum guard (review A8):** the linear approximation is poor far off-axis. Before building $J$ (and *only* for $J$), clamp $t_x/t_z$ and $t_y/t_z$ to $\pm 1.3\tan(\mathrm{fov}/2)$ and rescale by $t_z$. Without the clamp, near-edge Gaussians become giant streaks when you zoom in.

**Bug from the review (A5), restated as a rule:** keep $J$ as an explicit **2×3** in your own math type. Don't pad it to 3×3 in a column-major library. That padding is exactly where the transpose crept in.

### 1.5 Dilation, conic, radius

Write $\Sigma_{2D} = \begin{bmatrix} a & b\\ b & c\end{bmatrix}$. Then:

1. **Dilate** (low-pass filter; covariances of convolved Gaussians *add*): $a \mathrel{+}= 0.3,\; c \mathrel{+}= 0.3$.
2. $\det = ac - b^2$. If $\det \le 0$, cull.
3. **Conic** (inverse covariance, i.e. the quadratic form you evaluate per pixel):
   $\Sigma_{2D}^{-1} = \frac{1}{\det}\begin{bmatrix} c & -b \\ -b & a\end{bmatrix}$. Store $(A, B, C) = (c/\det,\, -b/\det,\, a/\det)$.
4. **Eigenvalues** (closed form for a symmetric 2×2): $m = \tfrac{a+c}{2}$, $\lambda_{1,2} = m \pm \sqrt{\max(0.1,\, m^2 - \det)}$. The reference floors the discriminant at 0.1.
5. **Radius**: $r = \lceil 3\sqrt{\lambda_1}\rceil$, i.e. 3σ along the major axis. Using a circle of the major-axis radius is conservative: it costs extra tiles for thin ellipses, and that waste is something you can measure.

All of steps 2–5 must use the **dilated** $(a,b,c)$ (review A6).

> **Optional experiment, opacity-aware radius:** the render kernel skips any contribution with $\alpha < 1/255$. Solving $o\,e^{-d^2/2} = 1/255$ gives $d = \sqrt{2\ln(255\,o)}\,\sigma$. That is 3.33σ for $o=1$, but only 2.26σ for $o = 0.05$. Using this per-Gaussian radius instead of a fixed 3σ shrinks L, often substantially. Measure L before and after: it's a lesson in "cost is ∝ L".

### 1.6 Screen rectangle in tile units

One function, used by preprocess, duplicate *and* the CPU oracle (review A4):

```
get_tile_rect(mean2d, r, tiles_x, tiles_y) -> (min, max):     // max is EXCLUSIVE
    min.x = clamp( int((mean2d.x - r)            / BLOCK), 0, tiles_x)
    min.y = clamp( int((mean2d.y - r)            / BLOCK), 0, tiles_y)
    max.x = clamp( int((mean2d.x + r + BLOCK - 1) / BLOCK), 0, tiles_x)
    max.y = clamp( int((mean2d.y + r + BLOCK - 1) / BLOCK), 0, tiles_y)
tiles = (max.x - min.x) * (max.y - min.y)     // ≥ 0 thanks to clamping
```

If `tiles == 0`, cull. This **replaces the centre-on-screen test** (review A7): a Gaussian whose centre is off-screen but whose footprint overlaps the image is kept.

### 1.7 Colour

SH0: `rgb = max(0, SH_C0 * f_dc + 0.5)`. This is precomputed at load, so preprocess just copies it (or the render kernel reads the scene buffer directly). Later, for SH degree 1–3, evaluate here using the view direction $\mathbf d = \mathrm{normalize}(\boldsymbol\mu - \mathbf c_{cam})$ in **world** space, with the standard real-SH constants from the reference implementation. This is why `CameraParams` carries the camera position.

### 1.8 Write outputs

`mean2d, depth = t_z, radius, conic_opacity = (A, B, C, opacity), tiles_touched`.

**Launch:** 1-D, 256 threads per block, `grid = ceil(N/256)`. **Skip the launch if N == 0**, since a zero-sized grid is a launch error.

---

## 2. Scan: `tiles_touched → offsets`

**Why a prefix sum?** It is *parallel memory allocation*. Each Gaussian needs a private, contiguous slice of the instance array, and its start is the sum of all earlier counts. An inclusive scan gives `offsets[i] = Σ_{j ≤ i} tiles_touched[j]`. Gaussian *i* writes slots `[offsets[i-1], offsets[i])` (with `offsets[-1] = 0`), and `L = offsets[N-1]`.

```
tiles_touched: [3, 0, 1, 2]        offsets (inclusive): [3, 3, 4, 6]       L = 6
slots:  G0→[0,3)  G1→∅  G2→[3,4)  G3→[4,6)
```

**Interview probe: "why not an `atomicAdd` on a global counter?"** It would work, but the slice order would be nondeterministic (frame-to-frame differences make debugging miserable), and every thread would contend on one address. The scan is deterministic and O(N) work, and CUB's implementation is a single decoupled-look-back pass.

Then read L back to the host (pinned, async + stream sync) and `ensure_capacity(L)` on the binning buffers ([03 §5](03-memory-and-data-movement.md#5-streams-and-synchronisation)). **If L == 0, skip to clearing the image.**

---

## 3. Duplicate with keys

```
if tiles_touched[idx] == 0: return                       // the A1 fix
off  = (idx == 0) ? 0 : offsets[idx - 1]
rect = get_tile_rect(mean2d[idx], radius[idx], ...)      // SAME function as preprocess
dbits = __float_as_uint(depth[idx])                      // the A3 fix
for ty in [rect.min.y, rect.max.y):
  for tx in [rect.min.x, rect.max.x):
     keys[off] = (uint64(ty * tiles_x + tx) << 32) | dbits
     vals[off] = idx
     off++
// Debug-only invariant: off == offsets[idx]
```

**Key layout:**

```
 63                      32 31                          0
┌──────────────────────────┬─────────────────────────────┐
│     tile id (row-major)  │  IEEE-754 bits of depth > 0 │
└──────────────────────────┴─────────────────────────────┘
  sort by tile first ──────▶  then front-to-back within the tile
```

**Why the float's raw bits sort correctly:** a positive IEEE-754 float is `[sign=0][exponent][mantissa]`. A larger exponent means a larger value, and for equal exponents a larger mantissa means a larger value. So comparing the bit patterns as unsigned integers gives the same order as comparing the floats. This **only holds for non-negative floats**. Negative floats would sort in reverse, but the near-plane cull guarantees $t_z > 0.2$.

**Debug-only invariant check:** after the loop, `off == offsets[idx]`. A device `assert` here catches any count/write mismatch immediately (it would have caught A4 on the first frame).

---

## 4. Radix sort

`cub::DeviceRadixSort::SortPairs(temp, temp_bytes, d_keys /*DoubleBuffer*/, d_vals, L, begin_bit = 0, end_bit = 32 + bit_width(T), stream)`

- **Why radix sort?** It does O(L·passes) work with no comparisons and its scatters are GPU-friendly. Comparison sorts are O(L log L) with branchy merges.
- **Why pass `end_bit`?** At 1080p, T = 120 × 68 = 8160 tiles fit in 13 bits, so only 45 of the 64 key bits are ever non-zero. LSD radix sort makes one pass per digit, so the pass count scales with `end_bit`. With 8-bit digits, that's 6 passes instead of 8, a free ~25%. (CUB's exact digit width depends on version and architecture; the proportionality is what matters.)
- **Stability:** LSD radix sort is stable. Instances with equal keys (same tile, bit-identical depth) keep their emission order, which is Gaussian-index order. That makes output deterministic.

---

## 5. Identify tile ranges

`tile_ranges` is cleared to `{0,0}` every frame (`cudaMemsetAsync`), so empty tiles have an empty range. One thread per instance:

```
i = global index; if i >= L: return
tile = sorted_keys[i] >> 32
if i == 0:                       ranges[tile].start = 0
else:
    prev = sorted_keys[i-1] >> 32
    if tile != prev:             ranges[prev].end = i; ranges[tile].start = i
if i == L - 1:                   ranges[tile].end = L
```

**Invariant (cheap to test on the host in Debug):** the non-empty ranges partition `[0, L)`, and `Σ_tiles (end − start) == L`.

---

## 6. Render: one block per tile, one thread per pixel

### 6.1 Compositing maths

For the sorted Gaussians $1..K$ covering a pixel, each contributes
$\alpha_i = \min\!\big(0.99,\; o_i \exp(-\tfrac12\,\mathbf d^\top \Sigma_{2D,i}^{-1}\mathbf d)\big)$ with $\mathbf d = \mathbf p - \boldsymbol\mu_{2D,i}$:

$$
C = \sum_{i=1}^{K} c_i\,\alpha_i\,T_i,\qquad T_i = \prod_{j<i}(1-\alpha_j),\qquad C_{\text{final}} = C + T_{K+1}\,c_{bg}
$$

**Front-to-back vs back-to-front (the CPU renderer uses the latter):** both compute the same $C$. Unroll the back-to-front "over" recurrence $C \leftarrow \alpha_i c_i + (1-\alpha_i) C$ and you get exactly the sum above. Front-to-back carries a running transmittance $T$ that **only decreases**. Once $T < 10^{-4}$, no remaining Gaussian can change the pixel by more than $10^{-4}$, so the thread can stop. Back-to-front can never stop early, because the most important (nearest) layer comes last. That is the whole reason the GPU path sorts front-to-back.

The quadratic form expands, using the stored conic $(A, B, C)$, to:

$$\text{power} = -\tfrac12\,(A\,d_x^2 + C\,d_y^2) - B\,d_x d_y$$

`if power > 0: skip` (numerical guard). `alpha = min(0.99, o * exp(power))`. `if alpha < 1/255: skip`.

The 0.99 clamp keeps $T$ from ever becoming exactly 0, which matters for the backward pass in training. The forward pass keeps it so the output matches what training saw.

### 6.2 Why shared-memory batching

Every Gaussian in a tile's range is evaluated by all 256 pixels of that tile. Without staging, that's 256 global reads of the same Gaussian data. With staging, the block cooperatively loads 256 Gaussians (one per thread) into shared memory, synchronises, then every thread loops over the batch from shared memory. That is roughly 256× fewer global loads.

Shared memory per block: 256 × (4 B id + 8 B xy + 16 B conic_opacity [+ 16 B rgb]) ≈ 7–11 KB, comfortably small.

### 6.3 Pseudocode shape

```
tile  = (blockIdx.x, blockIdx.y);  pix = tile * 16 + (threadIdx.x, threadIdx.y)
inside = pix.x < W && pix.y < H
done   = !inside                   // out-of-image threads DO NOT return (see 6.4)
range  = ranges[tile.y * tiles_x + tile.x]
T = 1; C = 0; n_contrib = 0

for batch_start in range.start .. range.end step 256:
    if __syncthreads_count(done) == 256: break        // whole block finished early
    k = batch_start + thread_rank
    if k < range.end:                                  // cooperative load
        id = point_list[k]
        s_xy[rank] = mean2d[id]; s_conic_o[rank] = conic_opacity[id]; s_rgb[rank] = rgb[id]
    __syncthreads()
    for j in 0 .. min(256, range.end - batch_start):
        if done: break
        d = pix - s_xy[j];  power = …;  if power > 0: continue
        alpha = min(0.99, s_conic_o[j].w * exp(power)); if alpha < 1/255: continue
        T_next = T * (1 - alpha)
        if T_next < 1e-4: done = true; break
        C += s_rgb[j] * alpha * T;  T = T_next;  n_contrib++

if inside: out[pix] = to_rgba8(C + T * bg)          // clamp to [0,1], ×255, ROUND (not truncate)
```

### 6.4 Two subtle correctness points (interview gold)

1. **No early `return` for pixels outside the image.** When W or H isn't a multiple of 16, edge tiles contain threads with no pixel. If they `return`, the remaining threads hit `__syncthreads()` with part of the block absent. That is undefined behaviour in general and a hang on some architectures. Those threads must stay, help load shared memory, and simply not write output. Test with an **801×601** window.
2. **The top-of-loop `__syncthreads_count` does two jobs.** It's the early-exit vote, *and* it's the barrier that stops the next batch's loads from overwriting shared memory that slower threads are still reading (a write-after-read hazard). Remove it while keeping the bottom `__syncthreads()`, and you get a race that `compute-sanitizer --tool racecheck` will report.

### 6.5 Output format

Write `uchar4` RGBA8 with `round(clamp(c, 0, 1) · 255)` and alpha = 255 (or `1 − T` if you want to visualise coverage). In Phase A the destination is `img_.image`. In Phase B it's the mapped PBO pointer, and the kernel doesn't know or care which. It receives an `ImageView`.

---

## 7. Debug views (build these; they are the learning payoff)

Add a `DebugView` enum to `RenderSettings`. Each view is a few lines in the render kernel or a tiny extra kernel:

| View | Shows | What you learn |
|---|---|---|
| `Centers` | a dot at each `mean2d` (today's output) | projection + intrinsics are right |
| `Ellipses` | the 1σ/3σ iso-contours of each conic | covariance projection (A5/A6/A8) is right; you can *see* the frustum-edge blow-up when the clamp is disabled |
| `TilesPerGaussian` | colour each Gaussian by `tiles_touched` | why large/near Gaussians dominate L |
| `GaussiansPerTile` | heatmap of `range.end − range.start` | load imbalance between tiles: the main performance limiter of the render kernel |
| `ContribCount` | per-pixel `n_contrib` heatmap | early termination at work; cost per pixel |
| `Depth` | normalised blended depth $\sum \alpha_i T_i z_i$ | sort order is right (A3) |
| `Alpha` | $1 - T_{final}$ | coverage, holes, background bleed |
| `CpuGpuDiff` | `abs(cpu − gpu) × gain` | localises any discrepancy to pixels/tiles instantly |

---

## 8. Failure-mode cheat sheet

| Symptom | Likely cause |
|---|---|
| Garbage or changing `L`; `cudaMalloc` asks for huge sizes | Culled Gaussians not writing `tiles_touched = 0` (A2) |
| Writes past the end of the key buffer; memcheck errors in duplicate | Count and write rectangles differ (A4) |
| Flickering/wrong layering inside one depth unit | Depth packed by value cast, not by bits (A3) |
| Ellipses correct at the image centre, wrong toward the edges | $J^\top\Sigma J$ instead of $J\Sigma J^\top$ (A5) |
| Huge streaks when zooming in near the frustum edge | Missing 1.3·tan(fov) Jacobian clamp (A8) |
| Splats pop at the screen border while orbiting | Culling by centre instead of by rectangle (A7) |
| Splats crawl in 1-px steps | Integer `mean2d` (A7) |
| Image upside-down after switching to interop or PNG dumps | Hidden reflection in the view matrix (B1) |
| Hang or garbage only for some window sizes | Early `return` before `__syncthreads` in edge tiles (§6.4) |
| Random speckles that move between runs | Missing barrier before the shared-memory reload (§6.4) |
| Image darker than the CPU reference | Truncation instead of rounding in the uint8 conversion; or the CPU accumulating in uint8 (E3) |

---

## Alternatives & trade-offs

### The core ordering strategy

| Option | How ordering is achieved | Pros | Cons |
|---|---|---|---|
| **Global sort of (tile, depth) keys** (chosen; the paper's method) | One radix sort over L instances | Exact match with training; each tile's list is contiguous; early termination | L ≫ N; sort is a major cost; per-tile ordering by *centre* depth can still "pop" when two Gaussians swap centre order |
| Global depth sort of N + graphics-pipeline blending | Sort N centres once; draw quads back-to-front with fixed-function blending | Sorts N, not L; portable (WebGL); ROPs do the blending | No early termination, heavy overdraw; 8-bit framebuffers band (use `RGBA16F`) |
| Per-tile local sort (shared-memory bitonic per tile) | Bin without depth, then sort each tile's list inside its block | No global sort; keys are 32-bit | Tiles with thousands of Gaussians overflow shared memory; load imbalance gets worse |
| Per-pixel resorting (StopThePop-style hierarchical sorting) | Refine the order per pixel with a small sorted window | Removes popping | Extra work per pixel; a research-grade refinement |
| Order-independent transparency (weighted blended OIT) | No sort; weight by depth | Cheapest | Approximate. Visibly wrong for the dense, opaque-ish Gaussians that make up real scenes |

### Sort key

| Option | Key size | Passes (8-bit digits) | Trade-off |
|---|---|---|---|
| **64-bit `[tile:32 \| depth bits:32]`, `end_bit = 32 + bits(T)`** (chosen) | 8 B | ~6 at 1080p | Exact |
| 32-bit `[tile:13–16 \| quantised depth:16–19]` | 4 B | 4 | Half the key memory and fewer passes; depth ties within a quantum → flicker |
| Segmented sort (sort only depth, within tile segments) | 4 B | varies | Needs segment offsets first; CUB's segmented sort is fast for many small segments, worse for a few huge ones |

### Footprint (radius) estimate

| Option | Tiles per Gaussian | Cost |
|---|---|---|
| **3σ circle of the major axis** (chosen; reference) | Most | Trivial |
| Opacity-aware radius ([§1.5](#15-dilation-conic-radius)) | Fewer for faint splats | One `log` per Gaussian |
| Exact ellipse–tile intersection (later optimised rasterizers test each candidate tile) | Fewest, especially for thin ellipses | Per-tile test in duplicate; the count and write passes must use the same test |

### Render kernel work decomposition

| Option | Pros | Cons |
|---|---|---|
| **1 thread = 1 pixel, 1 block = 1 tile** (chosen) | Simple; matches the reference | Threads whose pixels finish early idle until the block votes done; heavy tiles serialise |
| 1 thread = several pixels (e.g. 2×2) | Shared-memory loads amortised over more pixels; fewer blocks | More registers; divergence within a thread's pixels |
| Split heavy tiles across multiple blocks, then merge | Load balance for dense regions | A merge pass, because compositing is order-dependent: you must merge (C, T) pairs in order. The maths works: `C = C₁ + T₁·C₂`, `T = T₁·T₂` |
| Warp-per-tile with warp-level primitives | Fewer barriers (`__syncwarp`) | Only 32 pixels per tile, or more pixels per thread |

The (C, T) merge formula in the third row is worth remembering. It shows that front-to-back compositing is *associative*, which is why parallel and segmented compositing is possible at all.

### Library vs hand-written primitives

| Option | Pros | Cons |
|---|---|---|
| **CUB** (chosen) | Fastest available; explicit temp storage (fits the grow-only policy); stream parameter | Verbose two-call API (size query, then run) |
| Thrust (`thrust::sort_by_key`, `inclusive_scan`) | One-liners | **Allocates temporary storage internally on every call** unless you pass a custom allocator through an execution policy, which reintroduces hidden per-frame `cudaMalloc` |
| Hand-written scan/sort | The deepest learning about GPU algorithms | Weeks of work to match CUB; correctness risk. Do it as a separate exercise and compare against CUB in a test |

### Reuse vs write

| Option | Pros | Cons |
|---|---|---|
| **Write the rasterizer yourself, using the reference as a spec** (chosen) | The point of the project | Time |
| Vendor `diff-gaussian-rasterization` / gsplat kernels | Working on day one | PyTorch-oriented interfaces; you'd learn integration, not the algorithm |
| **Use the reference as an oracle only** (recommended in addition) | Render the same camera with the official renderer (Python) and diff the images | Needs a Python/PyTorch environment; see [06 Alternatives](06-roadmap-testing-profiling.md#alternatives--trade-offs) |
