# 05 — C++ Implementation Guide (signature level)

> **Mental model:** *a resource's lifetime should be a scope, not a pair of calls.* Every `cudaMalloc`/`cudaFree`, `glGen*`/`glDelete*` and map/unmap pair in this codebase becomes a constructor/destructor pair. Every function signature should show whether it **owns** (by value, move-only), **borrows** (a reference or view), or **shares** (`shared_ptr<const T>`).

This document gives declarations and the reasoning behind them, not full implementations. Where the semantics live in a body (the move constructor, `ensure_capacity`), I describe the body in prose.

**Language level:** move to **C++20** for host and device code (`CMAKE_CXX_STANDARD 20`, `CMAKE_CUDA_STANDARD 20`; nvcc ≥ 12.0 supports it with GCC ≥ 10). You get `std::span`, `std::bit_width` (for the sort's `end_bit`) and designated initialisers for the args structs.

---

## 1. RAII wrappers for CUDA resources

### 1.1 Error type and check macros (`render/cuda/cuda_check.h`)

```cpp
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* expr, const char* file, int line);
    cudaError_t code() const noexcept;
};

#define CUDA_CHECK(expr)   ::splat::cuda::check((expr), #expr, __FILE__, __LINE__)   // throws CudaError
#define CUDA_CHECK_LAST()  CUDA_CHECK(cudaGetLastError())                          // after every <<<>>>
#ifdef SPLAT_DEBUG_SYNC_KERNELS
#  define SPLAT_DEBUG_SYNC(stream) CUDA_CHECK(cudaStreamSynchronize(stream))        // localise async faults
#else
#  define SPLAT_DEBUG_SYNC(stream) ((void)0)
#endif
```

Why a function behind the macro: the macro only captures `#expr`, `__FILE__` and `__LINE__`. The logic lives in a normal, debuggable function. Throw (not `abort`, as the current `CUDA_CHECK` does) so that RAII unwinding releases the GL context and window cleanly at setup time ([02 §7](02-architecture.md#7-error-handling-policy)).

### 1.2 `DeviceBuffer<T>`: the owning type

```cpp
template <class T>
class DeviceBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "device memory holds bytes, not objects");
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t count);                       // exact allocation

    ~DeviceBuffer();                                               // cudaFree; logs errors, never throws

    DeviceBuffer(const DeviceBuffer&)            = delete;         // unique ownership
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;                   // steal + null the source
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;        // release own, steal, null source

    // Grow-only; CONTENTS ARE DISCARDED on growth. Sets size() = count.
    void ensure_capacity_discard(std::size_t count, float growth = 1.5f);

    void upload_async(std::span<const T> host, cudaStream_t s);    // requires host.size() <= capacity()
    void zero_async(cudaStream_t s);                               // cudaMemsetAsync over size()

    T*           data()           noexcept;
    const T*     data()     const noexcept;
    std::size_t  size()     const noexcept;
    std::size_t  capacity() const noexcept;
    DeviceSpan<T>       span()       noexcept;
    DeviceSpan<const T> span() const noexcept;

private:
    T*          ptr_      = nullptr;
    std::size_t size_     = 0;
    std::size_t capacity_ = 0;
};
```

**Why each piece is there (from first principles):**

- **Copy deleted:** a copy would duplicate the *pointer*, not the *memory*, so two destructors would free one allocation. That's review C3. Deleting copy turns this runtime double free into a compile error.
- **Move nulls the source:** the moved-from object's destructor *still runs*. If it still held the pointer, it would free memory the new owner is using. After the move, the source holds `nullptr`, and `cudaFree(nullptr)` is a no-op.
- **Move is `noexcept`:** `std::vector<DeviceBuffer<T>>` relocates elements on growth using `std::move_if_noexcept`. For a move-only type it moves regardless, but without `noexcept` it loses the strong exception guarantee. Moving pointers can't fail, so say so. (Interview follow-up: "when does `vector` copy instead of move during reallocation?" Answer: when the move constructor isn't `noexcept` *and* a copy constructor is available.)
- **Move-assignment** must release its *current* allocation before stealing. Handle self-move (`if (this != &other)`).
- **`static_assert(trivially_copyable)`:** device memory is raw bytes moved with `memcpy`. A type with a non-trivial copy or destructor would be silently mis-handled.
- **`ensure_capacity_discard`:** the name states the policy from [03 §2](03-memory-and-data-movement.md#2-allocation-policy-per-class). It's growth-only, it discards contents and it never shrinks. A reader can't mistake it for `std::vector::resize`, which preserves contents.

### 1.3 `DeviceSpan<T>`: the non-owning view kernels receive

```cpp
template <class T>
struct DeviceSpan {                         // aggregate, trivially copyable
    T*            data = nullptr;
    std::uint32_t size = 0;
    SPLAT_HD T& operator[](std::uint32_t i) const { return data[i]; }
};
static_assert(std::is_trivially_copyable_v<DeviceSpan<float>>);
```

**Why kernels take views, never owners:** kernel arguments are copied *bytewise* into the constant bank at launch. An owner can't be passed (its copy is deleted, so you get a compile error, which is good), and a view costs 12–16 bytes. The type system now encodes the rule "kernels borrow, the host owns".

### 1.4 Other wrappers (same pattern)

```cpp
class CudaStream { cudaStream_t s_{}; public: CudaStream();  /* cudaStreamNonBlocking */  ~CudaStream(); /* move-only */ cudaStream_t get() const noexcept; };
class CudaEvent  { cudaEvent_t  e_{}; public: CudaEvent();  ~CudaEvent();  void record(cudaStream_t); /* move-only */ };
template <class T>
class PinnedBuffer { /* cudaMallocHost / cudaFreeHost; std::span<T> view(); move-only */ };

class StageTimer {                          // per-stage GPU timings without host syncs mid-frame
public:
    void begin(Stage, cudaStream_t);        // records start event
    void end(Stage, cudaStream_t);          // records stop event
    StageTimings collect();                 // call after the frame's final sync: cudaEventElapsedTime
};
```

**Header hygiene trick:** `cudaStream_t` is just `CUstream_st*`. A header that only *passes* streams around can forward-declare `struct CUstream_st; using cudaStream_t = CUstream_st*;` instead of including `cuda_runtime.h`, which keeps CUDA headers out of host-only translation units.

---

## 2. The GPU backend facade (PIMPL)

```cpp
// render/cuda/gpu_renderer.h — includes NO CUDA headers
class GpuRenderer final : public IRenderer {
public:
    explicit GpuRenderer(const GpuRendererConfig& cfg);     // picks device (interop-compatible), creates stream
    ~GpuRenderer() override;                                // DECLARED here, DEFINED in the .cu
    GpuRenderer(GpuRenderer&&) noexcept;                    // = default in the .cu
    GpuRenderer& operator=(GpuRenderer&&) noexcept;         // = default in the .cu

    void       upload(const HostScene& scene) override;     // copies to device; retains nothing
    void       resize(Extent2D extent) override;            // image-class buffers only
    FrameStats render(const FrameRequest& req, const ImageView& target) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

**The classic PIMPL gotcha (interview staple):** `std::unique_ptr<Impl>` with an *incomplete* `Impl` compiles fine until something instantiates its destructor. If `~GpuRenderer()` is implicitly defined in the header, every TU that destroys a `GpuRenderer` instantiates `default_delete<Impl>`, which does `static_assert(sizeof(Impl) > 0)` and fails with "invalid application of sizeof to an incomplete type". The fix: *declare* the destructor (and the move operations) in the header and *define* them `= default` in the `.cu`, after `Impl` is complete.

**`Impl` layout:** member order is destruction order in reverse.

```cpp
struct GpuRenderer::Impl {
    CudaStream               stream;          // first declared → destroyed LAST (stream-ordered frees, if adopted later, need it alive)
    SceneBuffers             scene;           // immutable after upload()
    GeometryBuffers          geom;            // N-sized, overwritten per frame
    BinningBuffers           bin;             // L-sized, grow-only (+ CUB sort temp)
    ImageBuffers             img;             // pixel/tile-sized, realloc on resize
    PinnedBuffer<uint32_t>   num_instances{1};
    StageTimer               timer;
    Extent2D                 extent{};
    std::uint32_t            num_gaussians = 0;
};
```

Each `*Buffers` is a plain struct of `DeviceBuffer<T>` members. It needs **no destructor and no free function**, because the members' destructors run automatically. That deletes `free_scene` / `free_scratch` and their leaks (review C1) as a category.

---

## 3. Kernel interface: args structs + host launchers

```cpp
// render/cuda/kernels/kernel_args.h  (plain header, compiled by both g++ and nvcc)
struct PreprocessArgs {
    // inputs (scene)
    DeviceSpan<const Vec4f> mean_opacity;
    DeviceSpan<const Vec4f> cov3d_a;
    DeviceSpan<const Vec2f> cov3d_b;
    // per-frame uniforms, by value
    CameraParams   cam;
    RenderConsts   rc;            // near, dilation, tiles_x/y, …
    // outputs
    DeviceSpan<Vec2f>         mean2d;
    DeviceSpan<float>         depth;
    DeviceSpan<int>           radius;
    DeviceSpan<Vec4f>         conic_opacity;
    DeviceSpan<std::uint32_t> tiles_touched;
};
static_assert(std::is_trivially_copyable_v<PreprocessArgs>);
static_assert(sizeof(PreprocessArgs) <= 4096, "kernel parameter space");

void launch_preprocess(const PreprocessArgs&, cudaStream_t);     // defined in preprocess.cu
void launch_duplicate (const DuplicateArgs&,  cudaStream_t);
void launch_ranges    (const RangesArgs&,     cudaStream_t);
void launch_render    (const RenderArgs&,     cudaStream_t);
```

**Why one args struct per kernel instead of 18 positional parameters** (today's `project_gaussians`)? Positional `float*` parameters let you swap `d_depths` and `d_radii` with no compiler complaint. Named fields plus designated initialisers (`.depth = geom.depth.span()`) make every binding visible, and adding a field doesn't reshuffle every call site. The `launch_*` wrappers hide `<<<>>>` from host code and are the single place that computes the grid size, skips `N == 0` and calls `CUDA_CHECK_LAST()` / `SPLAT_DEBUG_SYNC()`.

---

## 4. Shared math for host and device

```cpp
// core/hd.h
#if defined(__CUDACC__)
#  define SPLAT_HD __host__ __device__ __forceinline__
#else
#  define SPLAT_HD inline
#endif

// core/math.h — tiny, POD, layout-explicit
struct alignas(8)  Vec2f { float x, y; };          // layout-compatible with CUDA float2 → 64-bit loads
struct Vec3f { float x, y, z; };                    // registers/temporaries only; never a buffer element
struct alignas(16) Vec4f { float x, y, z, w; };     // layout-compatible with CUDA float4 → 128-bit loads
static_assert(sizeof(Vec4f) == 16 && alignof(Vec4f) == 16);
struct Sym2  { float a, b, c; };                    // [[a,b],[b,c]]
struct Sym3  { float xx, xy, xz, yy, yz, zz; };
struct Mat3  { float m[9]; };                       // row-major, stated once
struct Mat23 { float m[6]; };                       // the Jacobian, explicitly 2×3 (review A5)

// core/projection.h
SPLAT_HD Sym3   cov3d_from_scale_rot(Vec3f scale, Vec4f quat_wxyz);
SPLAT_HD Vec3f  world_to_camera(const CameraParams&, Vec3f p_world);
SPLAT_HD Vec2f  project_mean(const CameraParams&, Vec3f t);
SPLAT_HD Mat23  projection_jacobian(const CameraParams&, Vec3f t);      // includes the 1.3·tan(fov) clamp
SPLAT_HD Sym2   project_cov(const CameraParams&, const Sym3& cov_w, Vec3f t);
SPLAT_HD bool   conic_and_radius(Sym2 cov2d_dilated, Vec3f& conic_out, int& radius_out);

// core/tile_rect.h
struct TileRect { int min_x, min_y, max_x, max_y; };                    // max EXCLUSIVE
SPLAT_HD TileRect get_tile_rect(Vec2f mean2d, int radius, int tiles_x, int tiles_y);
SPLAT_HD std::uint32_t tile_count(const TileRect&);

// core/compositing.h
SPLAT_HD float gaussian_alpha(Vec4f conic_opacity, Vec2f d, const RenderConsts&);  // returns 0 if skipped
```

**Why not Eigen or glm here?**

- **Eigen on device** is possible but heavy. Alignment and vectorisation assumptions differ between host and device, compile times balloon, and the CPU/GPU tests would still compare two differently configured libraries.
- **glm** works on both sides. But its column-major constructors are exactly what produced review A5, and it adds `GLM_FORCE_*` macro configuration to every TU.
- **~150 lines of explicit POD maths** that *you* wrote is the best possible learning artefact here. Every formula in [04](04-cuda-forward-pipeline.md) maps to one function. Keep Eigen for the app layer (camera manipulation, loader validation), and convert at exactly one point: `Camera::params() -> CameraParams`.

**Host vs device floating point won't be bit-identical,** even with shared source:

- nvcc fuses `a*b + c` into FMA by default (`--fmad=true`); x86 host code usually doesn't.
- `--use_fast_math` replaces `expf` with `__expf`.

So CPU-vs-GPU tests use tolerances ([06](06-roadmap-testing-profiling.md#2-testing-strategy)).

In `SPLAT_HD` code, call `sqrtf`/`expf`/`fminf` (available on both sides). Avoid `std::clamp` and `std::max` unless you compile with `--expt-relaxed-constexpr`.

---

## 5. Host-side scene types

```cpp
// scene/host_scene.h
struct HostScene {
    std::vector<Vec4f> mean_opacity;     // xyz world, w = sigmoid(opacity)
    std::vector<Vec4f> scale;            // exp(scale), w = 0
    std::vector<Vec4f> quat_wxyz;        // normalised; order in the NAME
    std::vector<float> sh;               // N × 16 × 3, coefficient-major, raw (DC NOT pre-baked to RGB)
    int                sh_degree = 0;    // inferred from the count of f_rest_* (0/9/24/45 → 0/1/2/3)
    SceneInfo          info;             // count, skipped, source format (see 07 §4)
    Aabb               bounds;           // for camera framing on load
    std::size_t size() const noexcept;
    void validate() const;               // all arrays equal length; throws otherwise
};

std::shared_ptr<const HostScene> load_ply(const std::filesystem::path&);   // throws on error
```

The `const` in `shared_ptr<const HostScene>` is the policy: *no one mutates the scene after loading*. Immutable shared data needs no locks and no defensive copies, and it can't be changed underneath a renderer.

---

## 6. OpenGL RAII and the interop lease

```cpp
// present/gl_resources.h
template <class Policy>
class GlHandle {                                  // move-only owner of one GL name
public:
    GlHandle() = default;
    explicit GlHandle(GLuint id) noexcept;
    ~GlHandle();                                  // if (id_) Policy::destroy(id_);
    GlHandle(GlHandle&&) noexcept;  GlHandle& operator=(GlHandle&&) noexcept;
    GlHandle(const GlHandle&) = delete;  GlHandle& operator=(const GlHandle&) = delete;
    GLuint get() const noexcept;
private:
    GLuint id_ = 0;
};
struct TexturePolicy { static void destroy(GLuint id) { glDeleteTextures(1, &id); } };
struct BufferPolicy  { static void destroy(GLuint id) { glDeleteBuffers(1, &id); } };
struct VaoPolicy     { static void destroy(GLuint id) { glDeleteVertexArrays(1, &id); } };
struct ProgramPolicy { static void destroy(GLuint id) { glDeleteProgram(id); } };
using GlTexture = GlHandle<TexturePolicy>;  using GlBuffer = GlHandle<BufferPolicy>; /* … */
```

*Why a policy struct rather than `template <auto DeleteFn>`:* with glad, `glDeleteTextures` is a macro for a function **pointer loaded at runtime**, not a compile-time constant. So it can't be a non-type template argument. A static member function that calls it at runtime can.

```cpp
// present/interop/cuda_gl_pbo.h
class CudaGlPbo {                                  // registration lifetime
public:
    CudaGlPbo(GLuint pbo, Extent2D);               // cudaGraphicsGLRegisterBuffer(…, WriteDiscard)
    ~CudaGlPbo();                                  // cudaGraphicsUnregisterResource (PBO + context must still be alive)
    CudaGlPbo(const CudaGlPbo&) = delete;          // non-copyable; movable if you need it
    [[nodiscard]] MappedPbo map(cudaStream_t);
};

class MappedPbo {                                  // one-frame LEASE
public:
    ImageView view() const noexcept;               // device pointer + pitch, valid only while *this lives
    ~MappedPbo();                                  // cudaGraphicsUnmapResources(…, stream)
    MappedPbo(MappedPbo&&) noexcept; /* copy deleted */
};
```

`[[nodiscard]]` on `map()` matters. `interop.map(stream);` as a bare statement would map and *immediately* unmap (the temporary dies at the semicolon). The attribute turns that silent no-op into a warning.

---

## 7. `Window`: the one non-movable owner

```cpp
class Window {
public:
    Window(const GlfwRuntime&, Extent2D, const char* title);   // throws if creation or GL load fails
    ~Window();
    Window(const Window&) = delete;  Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;       Window& operator=(Window&&) = delete;   // GLFW stores `this`
    Extent2D framebuffer_size() const;                          // HiDPI-correct size
    const InputState& input() const noexcept;                   // const member fn (today's isn't)
    …
};
```

GLFW callbacks find the C++ object through `glfwSetWindowUserPointer(win, this)`. Moving the object changes `this` but not what GLFW stored, so callbacks would write into a dead object. Making it non-movable is the honest encoding of that constraint. If it ever has to be relocatable, hold it in a `std::unique_ptr<Window>`: the heap address never changes.

---

## 8. CMake structure (target graph, not the full file)

```
splat_core           INTERFACE   core/*.h                            (no deps; usable by g++ and nvcc)
splat_scene          STATIC      scene/*.cpp                         → splat_core, tinyply, Eigen
splat_render_cpu     STATIC      render/cpu/*.cpp                    → splat_core, splat_scene
splat_present        STATIC      present/*.cpp                       → glad, glfw, OpenGL
splat_render_cuda    STATIC      render/cuda/**/*.cu                 → splat_core, CUDA::cudart   [if SPLATSCOPE_WITH_CUDA]
splat_interop        STATIC      present/interop/*.cpp               → splat_present, CUDA::cudart [if SPLATSCOPE_WITH_CUDA]
splatscope           EXECUTABLE  app/*.cpp                           → all of the above
splat_tests          EXECUTABLE  tests/**                            → doctest/GoogleTest           [if SPLATSCOPE_BUILD_TESTS]
```

Key settings and why:

| Setting | Value | Why |
|---|---|---|
| `project(... LANGUAGES C CXX)` + `enable_language(CUDA)` inside `if(SPLATSCOPE_WITH_CUDA)` | | CPU-only configure works (review F2) |
| Dependencies | `FetchContent_Declare(... GIT_TAG <pinned>)` for tinyply and the test framework; `find_package` for system GLFW/Eigen with a FetchContent fallback | A fresh clone builds (review F1); pinned tags make it reproducible |
| `CMAKE_CUDA_ARCHITECTURES` | default `native`, overridable cache var | `native` is convenient locally; CI/other machines need explicit values |
| CUDA flags, all configs | `-lineinfo` | Nsight Compute source correlation, no perf cost |
| CUDA flags, Debug | `-DSPLAT_DEBUG_SYNC_KERNELS` (not `-G` by default) | `-G` disables device optimisation: use only for cuda-gdb stepping |
| CUDA flags, optional | `-Xptxas=-warn-double-usage`, `-Xptxas=-v` | Catch accidental FP64 (review A9); print registers/spills per kernel |
| `CUDA_SEPARABLE_COMPILATION` | OFF | Not needed (review F3) |
| Warnings | `-Wall -Wextra -Wpedantic` (host); `-Xcompiler=-Wall` (nvcc host side) | Several review findings are warnings away |
| Sanitizers (host) | an `ASan+UBSan` preset for the CPU path and loader | Catches the host-side half of ownership bugs |
| `CMakePresets.json` | `debug`, `release`, `relwithdebinfo`, `cpu-only`, `asan` | One command per configuration; documents the supported builds |

---

## Alternatives & trade-offs

### Owning device memory

| Option | Pros | Cons |
|---|---|---|
| **Own `DeviceBuffer<T>`** (chosen) | Exact semantics you choose (grow-only, discard, no init); ~100 lines; you learn the rule of five | You maintain it |
| `std::unique_ptr<T[], CudaFreeDeleter>` | Zero code; standard move semantics | Carries no size or capacity; growth policy lives elsewhere |
| `thrust::device_vector<T>` | RAII, size, iterators, works with Thrust algorithms | **Copy constructor is a deep, implicit device copy** (an accidental by-value parameter copies megabytes); `resize` value-initialises with a fill kernel; temp allocations hidden in algorithms |
| RMM (`rmm::device_uvector`, `rmm::device_buffer`) | Uninitialised, stream-ordered, pluggable memory resources (pool, arena); production-grade | Large dependency (RAPIDS) |

### Error handling

| Option | Pros | Cons |
|---|---|---|
| **Exceptions at setup; checked-but-cheap in the frame loop** (chosen) | RAII unwinding releases GL/CUDA cleanly; error reporting stays out of every signature | Exceptions can't cross into device code (irrelevant: device code reports via sticky errors) |
| `std::expected<T, Error>` (C++23) or `tl::expected` (C++17/20) | Errors visible in signatures; no hidden control flow | Every call site must propagate; noisy for "should never happen" CUDA failures |
| `abort()` on error (today) | Simplest; core dump at the failure | No cleanup, no message to the user for ordinary errors such as a bad file path |

### Passing per-frame uniforms to kernels

| Option | Pros | Cons |
|---|---|---|
| **By value in an args struct** (chosen) | No copy call; broadcast from the constant bank; changes are automatically ordered with the launch | 4 KB limit (32 KB on newer setups); copied per launch (tiny) |
| `__constant__` symbol + `cudaMemcpyToSymbolAsync` | Shared by several kernels without repeating the parameter | Global state in the `.cu`; you must order the copy with the stream yourself; the kernel can't be launched twice concurrently with different values |
| Device buffer + pointer (today) | Arbitrary size | An extra copy and an extra indirection |

### Hiding CUDA from the app

| Option | Pros | Cons |
|---|---|---|
| **PIMPL concrete class** (chosen) | A concrete `GpuRenderer` type: GPU-only methods (debug views, stats) callable without casting | Out-of-line special members; one heap allocation |
| Interface + factory: `std::unique_ptr<IRenderer> make_gpu_renderer(cfg)`; the class lives entirely in a `.cu` | Less boilerplate: no `Impl`, no out-of-line destructor dance, CUDA types allowed in the class | Only the interface is visible; GPU-specific features must go through the interface or `dynamic_cast` |

Both are fine. If you don't expect GPU-only methods, **the factory is simpler**: it gets the same compile-time isolation with less code.

### Device-side maths

| Option | Pros | Cons |
|---|---|---|
| **Own `SPLAT_HD` POD header** (chosen) | Exactly the operations needed; explicit layout and row/column order; identical source on CPU and GPU | You write and test it (~150 lines) |
| glm (`GLM_FORCE_CUDA`) | Mature; GLSL-like syntax familiar from your shader work | Column-major constructor traps (review A5); global macro configuration |
| CUDA `float3`/`float4` + `helper_math.h` (from the CUDA samples) | Native vector types; the operators the samples use | Not a header CUDA guarantees; it's copied from the samples. No matrix types |
| Eigen in device code | Same library as the host | Heavy compile times; partial device support; alignment pitfalls |

### GL handle ownership

| Option | Pros | Cons |
|---|---|---|
| **`GlHandle<Policy>`** (chosen) | Clear, typed, move-only | One small template |
| `std::unique_ptr` with a custom deleter | Standard | `GLuint` isn't a pointer: the deleter must define a `pointer` type that satisfies *NullablePointer*. Doable, but more obscure than a 30-line class |
| A wrapper library (globjects, Magnum) | Complete | Large dependency for six object types |

### Test framework

doctest (single header, fastest compile) vs GoogleTest (most common in industry; parameterised tests, death tests) vs Catch2 (BDD style). **GoogleTest** if interview familiarity matters, **doctest** if compile time does. Either way, wrap GPU tests in a fixture that skips cleanly when no CUDA device is present.
