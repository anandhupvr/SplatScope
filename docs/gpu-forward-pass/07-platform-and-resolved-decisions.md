# 07 — Your Platform and the Resolved Decisions

> **Mental model:** *a 4 GB, ~30–50 W laptop GPU behind an Intel iGPU changes three things: memory becomes a budget you plan for, the display path goes through another GPU, and timings drift with temperature.* The algorithm doesn't change. The defaults and the order of work do.

Decisions you made, and what each one changes:

| Question | Your answer | Consequence in the plan |
|---|---|---|
| Machine | MSI GF63 Thin, GTX 1650 Max-Q, latest CUDA | §1: sm_75, 4 GB VRAM, Optimus hybrid graphics. Phase A (readback) is the **default**; Phase B (interop) needs the GL context on the NVIDIA GPU |
| Debug controls | Keyboard only; ImGui only if truly needed | §2: hotkeys, stats in the window title, PPM/PNG dump. ImGui behind an OFF-by-default CMake option |
| Colour | **SH degree 3 required** (all 45 `f_rest` + 3 `f_dc`) | [04 §1.7](04-cuda-forward-pipeline.md#17-colour-spherical-harmonics-degree-03) has the full spec; §3 covers the memory impact; it becomes milestone M6 instead of an optional extension |
| Scene source | Official 3DGS training PLY now, other exporters later | §4: `HostScene` is the stable contract; one loader now, no plugin system |

---

## 1. The GPU: GTX 1650 Max-Q (Turing, sm_75)

### 1.1 Numbers that shape the design

Verify these on your unit with `nvidia-smi` and a startup log of `cudaGetDeviceProperties` + `cudaMemGetInfo`.

| Property | Value | Design consequence |
|---|---|---|
| Architecture / compute capability | Turing TU117, **7.5** | `CMAKE_CUDA_ARCHITECTURES 75` (explicit, so builds don't depend on the machine) |
| SMs / CUDA cores | 16 / 1024 | Preprocess with N in the millions saturates it easily; a 720p frame is 3600 tile blocks (≈225 per SM), so the render kernel has plenty of parallelism too |
| **Max resident threads per SM** | **1024** (32 warps). Turing is lower than Volta/Ampere's 2048 | A 256-thread render block → at most 4 blocks per SM. Full occupancy needs ≤ **64 registers/thread** (64K regs / 1024 threads). Check with `-Xptxas -v` |
| Shared memory | Up to 64 KB per block (configurable L1/shared split) | The ~7–11 KB per render block from [04 §6.2](04-cuda-forward-pipeline.md#62-why-shared-memory-batching) is far from a limit |
| VRAM | **4 GB** (≈ 3.5 GB usable after the driver context; measure it) | §3: memory budget, fp16 SH option, an L cap from `cudaMemGetInfo` |
| Memory bandwidth | ~128 GB/s (GDDR5) or ~192 GB/s (GDDR6), depending on the variant | Preprocess is bandwidth-bound once SH3 is on (§3.3) |
| FP64 throughput | 1/32 of FP32 | Review A9 (`1.0 / z`) is a real cost here |
| Power limit | Max-Q: low tens of watts | Clocks drift with temperature (§1.4) |

**CUDA version:** recent toolkits (the 13.x line) dropped offline compilation for architectures older than Turing. sm_75 is still supported, but it is now the *oldest* one supported. Pin the toolkit version you develop with (record it in the README and the CMake presets), so a future major release can't silently drop your GPU.

### 1.2 Hybrid graphics (Optimus): who owns the screen

On the GF63 Thin, the internal panel is wired to the **Intel iGPU**, and there's no MUX switch. By default the GLFW/GL context is created on the Intel GPU. CUDA always runs on the NVIDIA GPU. So:

- **Phase A (readback)** always works: CUDA renders into its own buffer, you copy to the host, and GL (on Intel) uploads it as a texture.
- **Phase B (interop)** works **only if the GL context is on the NVIDIA GPU too**:
  - **PRIME render offload (recommended):** run `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./splatscope scene.ply`, or your distribution's `prime-run` wrapper. This needs the proprietary NVIDIA driver, and the system's PRIME profile must be *on-demand* (Ubuntu: `prime-select query` → `on-demand`), not *intel*-only.
  - **Global NVIDIA mode** (Ubuntu `prime-select nvidia`, or the equivalent in your distribution's GPU switcher, then log out and back in) puts the whole X session on the dGPU. Interop works without environment variables, at the cost of battery life and heat for everything else running.
  - **Wayland sessions:** those two variables select the *GLX* vendor, so they only apply when GLFW uses X11. GLFW ≥ 3.4 picks Wayland (EGL) by default on a Wayland session. The simplest fix is `glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11)` before `glfwInit()`, which runs the window through XWayland where GLX offload works. Make it a command-line flag (`--x11`) rather than unconditional.
  - **Verify, don't assume:** `glxinfo -B` run with the same environment shows which GPU renders (`OpenGL renderer string`). The app's startup log of `GL_RENDERER` must say NVIDIA for Phase B to be enabled.

**Runtime decision (do this, don't hard-code):** at startup, log `glGetString(GL_VENDOR/GL_RENDERER)`, then call `cudaGLGetDevices(..., cudaGLDeviceListAll)`. Enable Phase B only if it returns the CUDA device you're using. Otherwise fall back to Phase A, with one log line explaining why. A wrong guess here shows up as a registration error or a black window that looks like a renderer bug.

**Set your expectations for interop on this laptop:** even with offload, the NVIDIA driver copies every finished frame to the Intel GPU for scan-out. Interop removes *your* two explicit copies, but the frame still crosses PCIe once. At a 1280×720 render size the Phase A readback is ~3.7 MB each way, roughly half a millisecond in total. That's not the bottleneck. **Conclusion: Phase A is the default on this machine. Phase B is a learning milestone, not a performance requirement.**

### 1.3 Default render size

Use **1280×720**, not the panel's 1920×1080. You get 3600 tiles instead of 8160, so L and the render cost are roughly halved, which leaves headroom for SH3 on 4 GB. Resize support (from [02 §5](02-architecture.md#5-per-frame-control-flow)) still lets you go full-screen.

### 1.4 Measuring on a thermally limited laptop

- Plug into AC power and use the vendor's performance mode. Battery mode caps the GPU further.
- Warm up for ~100 frames. Report the **median and p95 over ≥ 300 frames**, never a single frame or a short mean.
- Compare A/B variants back to back in the same session, ideally by toggling at runtime, not across reboots.
- On Linux the dGPU may be powered down when idle, so the first CUDA call can take hundreds of milliseconds. Never include initialisation in frame timings.
- `glfwSwapInterval(0)` while measuring ([03 §5](03-memory-and-data-movement.md#5-streams-and-synchronisation)). With PRIME offload the driver may still throttle presentation, so the GPU stage timings from CUDA events are the numbers to trust, not FPS.

### 1.5 Linux toolchain checklist

These are the Linux-specific failures you're likely to hit, in roughly the order you'll hit them:

| Symptom | Cause | Fix |
|---|---|---|
| `CMake 3.30 or higher is required` | `CMakeLists.txt:1` asks for 3.30, but distribution packages often lag (Ubuntu 24.04 ships 3.28) | Nothing used needs 3.30; `CMAKE_CUDA_ARCHITECTURES native` needs **3.24**. Lower the minimum, or install a newer CMake from Kitware's apt repository or `pip install cmake` |
| `unsupported GNU version! gcc versions later than N are not supported` from nvcc | Rolling or recent distributions ship a newer GCC than your nvcc accepts | Install a supported GCC alongside and set `-DCMAKE_CUDA_HOST_COMPILER=g++-<supported>` (put it in the CMake preset). Don't use `-allow-unsupported-compiler` except as a last resort |
| `nvcc` not found / CMake can't find CUDA | Toolkit installed under `/usr/local/cuda-<ver>` but not on `PATH` | `export PATH=/usr/local/cuda/bin:$PATH` or `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` in the preset |
| `ncu` reports `ERR_NVGPUCTRPERM` | GPU performance counters are restricted to root by default | Run `ncu` with `sudo`, or set `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in a file under `/etc/modprobe.d/`, rebuild the initramfs and reboot |
| `cudaGetDeviceCount` returns 0 or "no CUDA-capable device" | Nouveau driver loaded, or the proprietary driver isn't installed or loaded | `nvidia-smi` must list the GTX 1650; blacklist nouveau and install the distribution's NVIDIA driver package |
| Driver/toolkit mismatch ("driver version is insufficient for CUDA runtime version") | Toolkit newer than the driver supports | Upgrade the driver, or install the toolkit version matching it (`nvidia-smi` shows the maximum CUDA version the driver supports) |
| First frame or first CUDA call takes ~0.5 s | Runtime power management woke the powered-down dGPU | Expected; exclude initialisation from timings |

Development packages the build needs (Debian/Ubuntu names; other distributions are similar): `libglfw3-dev`, `libeigen3-dev`, `pkg-config`, plus the CUDA toolkit. glm is no longer needed once the shared maths header replaces it ([05 §4](05-cpp-implementation-guide.md#4-shared-math-for-host-and-device)), and tinyply comes in through FetchContent.

---

## 2. Debug controls: keyboard only, ImGui optional

Everything below needs only GLFW, which is already a dependency.

| Key | Action | Edge or level |
|---|---|---|
| Left drag / scroll / right drag | Orbit / zoom / pan | level (held) |
| `G` | Toggle GPU ⇄ CPU renderer (same camera, same settings) | edge |
| `1`–`8` | Debug view (`Final`, `Centers`, `Ellipses`, `TilesPerGaussian`, `GaussiansPerTile`, `ContribCount`, `Depth`, `Alpha`) | edge |
| `H` | Cycle the active SH degree 0 → 1 → 2 → 3 | edge |
| `U` | Flip `world_up` (scene convention, [02 §6](02-architecture.md#6-coordinate-conventions-single-source-of-truth)) | edge |
| `R` | Reset the camera to frame the scene AABB | edge |
| `P` | Save the current frame (PPM: a 10-line writer, no dependency; or PNG via the single-header `stb_image_write`) | edge |
| `T` | Print the `FrameStats` of the last frame to the console | edge |
| `V` | Toggle vsync | edge |

The window title shows `N, visible, L, GPU ms, FPS`, updated about twice a second with `glfwSetWindowTitle`.

**Edge vs level, the retention hook:** orbiting reads the *current state* of the mouse button (level). A toggle must fire on the *transition* from up to down (edge). If you implement toggles by polling `glfwGetKey(...) == GLFW_PRESS`, holding `G` for 100 ms flips the renderer six times at 60 FPS, and it looks random. So `InputState` gets a "pressed this frame" set, filled by the key callback and cleared in `poll_events()`. That's the same per-frame reset you already do for `mouse_dx` and `scroll_dy`.

**When ImGui would actually be worth it:** for continuous parameters you want to sweep (dilation, near plane, the α threshold) or live timing plots. Keyboard `+`/`-` steps cover the first; console output covers the second. Keep a `SPLATSCOPE_WITH_IMGUI` option, OFF by default, and don't add it until a keyboard workflow actually gets in your way.

---

## 3. Memory budget on a 4 GB GPU

### 3.1 Bytes per Gaussian with SH degree 3

| Storage | Scene buffers | Per-frame (preprocess outputs) | Total |
|---|---|---|---|
| **fp32 SH** | 16 (mean+opacity) + 24 (Σ) + 192 (SH) = **232** | 56 | **288 B** |
| fp16 SH rest (DC kept fp32) | 16 + 24 + 12 (DC) + 92 (45 × `half`, padded to 4 B) = **144** | 56 | **200 B** |

### 3.2 Budget at 1280×720 (3600 tiles)

These use L = 4·N as a typical orbit view at 720p. **Zooming in raises L sharply**, so measure your own scenes and read L from the title bar. Binning is 24 B × L × 1.5 (grow-only headroom).

| Scene size | fp32 SH: Gaussians + binning | fp16 SH: Gaussians + binning |
|---|---|---|
| 1M | 288 MB + 144 MB ≈ **0.4 GB** | 200 MB + 144 MB ≈ **0.35 GB** |
| 3M | 864 MB + 432 MB ≈ **1.3 GB** | 600 MB + 432 MB ≈ **1.0 GB** |
| 6M | 1.73 GB + 864 MB ≈ **2.6 GB** | 1.2 GB + 864 MB ≈ **2.1 GB** |

Add the CUB temp storage, the image buffers (~4 MB) and a **transient** 32 B × N during upload (scale + quaternion, freed after `build_cov3d`). The official training outputs for common benchmark scenes range from about one to six million Gaussians, so **fp32 fits typical scenes, but a 6M scene plus a close-up view gets close to the ~3.5 GB usable**.

**Decisions:**

1. **Start with fp32 SH.** The CPU oracle comparison is exact (to float tolerance), and the smaller benchmark scenes fit comfortably.
2. **Set the L cap at startup** from `cudaMemGetInfo` (e.g. 80% of free memory after the scene upload). Over the cap, report it in `FrameStats` and render a degraded frame ([03 §2](03-memory-and-data-movement.md#2-allocation-policy-per-class)). Don't let the process die from OOM.
3. **Add an fp16 storage option for `f_rest` as its own small milestone**, when a scene doesn't fit or when profiling shows SH bandwidth dominates (§3.3). It's a load-time switch (`ShStorage::F32 | F16`). The DoD: the degree-3 image matches fp32 within max |Δ| ≤ 1/255.

### 3.3 Bandwidth: why SH dominates preprocess

With N_visible = 3M and fp32 SH, preprocess reads 3M × 192 B ≈ **576 MB of SH per frame**. At 128–192 GB/s that's **~3–4.5 ms for SH alone**, before Σ, means and writes. fp16 halves it. That's why SH is evaluated **only for Gaussians that survive culling**, and why fp16 storage is the first optimisation to reach for on this GPU. A realistic target for a few-million-Gaussian scene at 720p on this laptop is **interactive (tens of FPS)**. Measure rather than trust this estimate.

### 3.4 Host RAM: the loader is the peak

An official PLY stores **62 floats per Gaussian**: xyz, normals (unused zeros), 3 DC, 45 rest, opacity, 3 scale, 4 rotation. That's 248 B, so a 6M scene is a **~1.5 GB file**. Today's loader holds several copies at once:

- the whole file in memory (`preload_into_memory = true`);
- tinyply's per-property buffers;
- the decoded vectors;
- `filter_valid`'s full copy.

That peaks at around 4× the file size, which is too much for a laptop with 8–16 GB of RAM. Fixes, in order of payoff:

1. Stream from `std::ifstream` (the existing flag set to `false`).
2. Decode each property group into `HostScene` and **release its tinyply buffer immediately**.
3. Filter invalid Gaussians **in place** (stable compaction over all SoA arrays at once) instead of copying into a new cloud.

---

## 4. Scene formats: one loader now, one contract forever

**Principle:** `HostScene` is the **contract** between "whatever is on disk" and the renderers, an *anti-corruption layer*. Everything downstream sees exactly this:

- activated values (`exp` scale, `sigmoid` opacity);
- normalised quaternions in **wxyz**;
- SH in **coefficient-major** order, with raw DC (not pre-baked to RGB, per [04 §1.7](04-cuda-forward-pipeline.md#17-colour-spherical-harmonics-degree-03));
- `sh_degree` stated explicitly;
- every Gaussian finite and valid.

Format quirks never leak past the loader.

**Now:** one function, `load_3dgs_ply(path) -> std::shared_ptr<const HostScene>`.

- Detect by property names (`x`, `f_dc_0`, `opacity`, `scale_0`, `rot_0`, and the count of `f_rest_*`).
- On mismatch, **fail loudly**, listing the missing or unexpected properties.
- Ignore `nx/ny/nz`.
- Fill a `SceneInfo { count, skipped, sh_degree, source_format, bounds }` and print it at startup.

**Later**, when a second format arrives, add a second `load_*` function and dispatch on extension or header. Don't build a registry or plugin system until then (YAGNI). Exported formats differ from the training PLY in exactly the ways the contract already pins down, so each new loader is a pure conversion function:

- values already activated, or not;
- quantised fields (8-bit colours, packed quaternions);
- reduced or missing SH;
- chunked compression;
- different axis conventions.

Its tests are "load file X, compare against `HostScene` Y".

---

## 5. Roadmap changes

This changes the milestones in [06](06-roadmap-testing-profiling.md#1-milestones):

- **M0** adds startup device logging (GPU name, compute capability, free/total VRAM, `GL_RENDERER`) and the edge-triggered key handling (§2).
- **M5** makes Phase B **conditional**: it is done when the runtime selection from §1.2 works both ways (offload on → interop; offload off → readback with a log line).
- **M6 is now SH degree 0–3 (required).** It covers:
  - the loader reorder and degree detection;
  - a shared `SPLAT_HD eval_sh(degree, dir, coeffs)` used by the CPU oracle and the kernel;
  - the `H` hotkey.

  DoD: per-Gaussian GPU `rgb` matches the CPU within 1e-5 at all four degrees; a unit test checks the `f_rest` reorder against a hand-built PLY; the L cap from §3.2 is in place.
- **M6b (when needed):** fp16 `f_rest` storage (§3.2).
- **M7:** the former M6 extension list.

---

## Alternatives & trade-offs

### SH layout in device memory

| Option | Layout | Pros | Cons |
|---|---|---|---|
| **Per-Gaussian, coefficient-major `[N][16] × RGB`** (chosen; reference layout) | 192 contiguous bytes per Gaussian | Matches the reference, so it's easy to compare; one pointer + offset per Gaussian | A warp's first load touches 32 lines 192 B apart; later loads hit L1. Fine in practice, so measure before changing it |
| SoA by coefficient (`k_j[N]` as 16 arrays) | Coefficient j of all Gaussians contiguous | Perfectly coalesced per load instruction | 16 pointers in the args struct; awkward to stage |
| Channel-major as in the file (`[N][3][15]`) | No reorder at load | No loader code | Three scattered reads per coefficient in the kernel; a convention that leaks past the loader (§4) |

### Where SH is evaluated

| Option | Correctness | Cost |
|---|---|---|
| **Per Gaussian in preprocess, centre direction** (chosen; paper) | Matches training | ~1 evaluation per visible Gaussian |
| Per pixel, with each pixel's own ray direction | Slightly more accurate for huge, close Gaussians | ~256× more evaluations, plus 192 B per Gaussian in shared memory; *doesn't* match what training optimised |
| Precomputed RGB on the CPU per frame | — | Uploads N × 12 B per frame and wastes the GPU. Only useful as a debugging step |

### Presentation on Optimus

| Option | Pros | Cons |
|---|---|---|
| **Phase A on the iGPU's GL** (default) | Always works; no environment setup | Two explicit 3.7 MB copies per frame (≈0.5 ms) |
| Phase B with PRIME render offload (or global NVIDIA mode) | Teaches interop; no explicit copies | Needs environment variables (and X11/XWayland under GLFW); the driver still copies to the iGPU for display |
| Headless rendering (EGL on the NVIDIA device, write images) | No window system; perfect for automated golden tests | Not interactive. Useful as the test driver from [02 §9.4](02-architecture.md#94-presentation-api) |

### Debug UI

| Option | Pros | Cons |
|---|---|---|
| **Hotkeys + window title + console** (chosen) | No dependency; forces you to name every state explicitly | No sliders or plots |
| Dear ImGui (optional, OFF) | Sliders, plots, tables in an afternoon | One more dependency and another render pass in the GL code |
| A config file reloaded on a hotkey (`settings.ini` → `RenderSettings`) | Continuous parameters without a UI library | You write a tiny parser; edits aren't live while you drag |
