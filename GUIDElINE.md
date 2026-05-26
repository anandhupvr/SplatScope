# C++ Project Guidelines — gs-viewer

A practical reference for code style, project structure, and tooling for the CPU 3D Gaussian Splatting viewer (and any C++ project of similar scope).

This document is the single source of truth for conventions. When in doubt, check here. When tools disagree with this document, update one of them so they match.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Naming Conventions](#2-naming-conventions)
3. [File Conventions](#3-file-conventions)
4. [Namespaces](#4-namespaces)
5. [Class Design](#5-class-design)
6. [Modern C++ Features](#6-modern-c-features)
7. [Formatting](#7-formatting)
8. [CMake](#8-cmake)
9. [Comments and Documentation](#9-comments-and-documentation)
10. [VS Code Setup](#10-vs-code-setup)
11. [Quick Reference Card](#11-quick-reference-card)
12. [References](#12-references)

---

## 1. Project Structure

```
gs-viewer/
  CMakeLists.txt
  cmake/                # CMake helpers (find modules, etc.)
  third_party/          # Optional: FetchContent destination
  src/
    main.cpp            # Wires it all together
    scene/
      gaussian_cloud.h
      ply_loader.h
      ply_loader.cpp
      ply_loader_test.cpp
    camera/
      camera.h
      camera.cpp
    splat/              # Rendering math primitives (reusable)
      projection.h
      projection.cpp
      projection_test.cpp
      covariance.h
      covariance.cpp
      covariance_test.cpp
      rasterize.h
      rasterize.cpp
    renderer/
      i_renderer.h      # Abstract interface
      cpu_renderer.h
      cpu_renderer.cpp
      framebuffer.h
    viewer/
      window.h
      window.cpp
      display.h
      display.cpp
      controls.h
      controls.cpp
  assets/               # Test PLYs and small data
  scripts/              # Python helpers (PLY inspector, etc.)
  docs/                 # Notes, derivations, this file
  .clang-format
  .clang-tidy
  .gitignore
  README.md
```

**Key rules:**
- Group by subsystem (`scene/`, `camera/`, `splat/`, `renderer/`, `viewer/`)
- Tests colocated with code: `foo.cpp` + `foo_test.cpp` in the same directory
- Headers and sources in the same folders (not split `include/`/`src/`)
- One include root: `src/`. Use prefixed includes: `#include "splat/projection.h"`

**Math files (`splat/*`) should be dependency-pure** — depend only on Eigen and standard headers, never on other project files. This makes them reusable in future projects.

---

## 2. Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Types (classes, structs, enums) | `UpperCamelCase` | `GaussianCloud`, `CpuRenderer` |
| Functions / methods | `snake_case` | `project_to_screen`, `load_ply` |
| Variables (local + parameters) | `snake_case` | `view_matrix`, `pixel_index` |
| Member variables | `snake_case_` (trailing underscore) | `cloud_`, `framebuffer_` |
| Constants / constexpr | `kUpperCamelCase` | `kMaxDepth`, `kNearPlane` |
| Enum values | `UpperCamelCase` | `enum class Mode { Cpu, Gpu };` |
| Namespaces | `lowercase` (short) | `namespace gs`, `namespace splat` |
| File names | `snake_case.h/cpp` | `cpu_renderer.cpp` |
| Macros (use sparingly) | `UPPER_SNAKE_CASE` | `GS_ASSERT(x)` |
| Template parameters | `UpperCamelCase` | `template<typename Scalar>` |

### Why these specific choices

- **Types `UpperCamelCase`, functions `snake_case`**: matches the C++ standard library style (`std::vector`, `std::sort`). Reading code that mixes STL and project code looks consistent.
- **Trailing underscore on members**: unambiguous, legal, no `m_` prefix ugliness, no shadowing with parameters. Leading underscore is reserved in some contexts — don't use it.
- **`kCamelCase` for constants**: comes from Google. Easy to spot. Alternative `UPPER_SNAKE` is also acceptable but looks like a macro.
- **`enum class` always, never plain `enum`**: scoped, no implicit conversions, no name leakage.

### Interface naming

For abstract base classes, two options — pick one and stick to it:
- **`Renderer` + `CpuRenderer` + `GpuRenderer`** (no prefix; modern C++ style, what Google/Core Guidelines recommend)
- **`IRenderer` + `CpuRenderer` + `GpuRenderer`** (the `I` prefix is clear but anachronistic)

This project uses **no prefix**.

---

## 3. File Conventions

### Header / source extensions
- Header: `.h`
- Source: `.cpp`
- CUDA: `.cu` / `.cuh`
- Shaders: `.vert`, `.frag`, `.comp`

### One class per file
Match the class name in snake_case. `CpuRenderer` → `cpu_renderer.h` / `cpu_renderer.cpp`. Small related types (a `Color` enum next to a `Color` struct) can share a file.

### Include guards
Use `#pragma once`. Every header. No exceptions.

```cpp
#pragma once

namespace gs::splat {
// ...
}  // namespace gs::splat
```

### Include order
Strictly enforced (clang-format does this automatically):

```cpp
// 1. The matching header (in .cpp files only)
#include "splat/projection.h"

// 2. C system headers
#include <cmath>
#include <cstdint>

// 3. C++ standard library
#include <algorithm>
#include <vector>

// 4. Third-party libraries
#include <Eigen/Dense>
#include <GLFW/glfw3.h>

// 5. Project headers
#include "scene/gaussian_cloud.h"
#include "camera/camera.h"
```

The "matching header first" rule catches missing includes — if `projection.h` forgets to include `<cmath>`, the `.cpp` won't paper over the bug via other includes.

### Forward declarations
Prefer forward declarations over `#include` in headers when possible. Reduces compile time and breaks dependency cycles.

```cpp
// camera.h
#pragma once

namespace gs {

class Frustum;  // forward declaration — no #include "frustum.h"

class Camera {
    const Frustum& frustum() const;
};

}  // namespace gs
```

`#include "frustum.h"` goes in `camera.cpp`.

---

## 4. Namespaces

### One project, one root namespace
`gs`. Lowercase. Short. No `Gs`, no `GS`.

### Nested namespaces for subsystems
```cpp
namespace gs::splat {
    Eigen::Matrix2f propagate_covariance_2d(...);
}  // namespace gs::splat
```

Use C++17 nested-namespace syntax (`namespace gs::splat`), not the old nested-braces style.

### Rules
- **Never `using namespace std;` in headers.** Pollutes every file that includes the header.
- **`using namespace ...;` inside a `.cpp` is fine** if it improves readability.
- **`using namespace ...;` inside a function is best** — narrowest scope.
- **Always close namespaces with a comment**: `}  // namespace gs::splat`

---

## 5. Class Design

### Member access order
```cpp
class CpuRenderer : public Renderer {
 public:
    // Constructors, destructor
    CpuRenderer();
    ~CpuRenderer() override = default;

    // Public methods
    void render(const Scene& scene, const Camera& cam, Framebuffer& out) override;

 private:
    // Private methods
    void project_all(...);
    void rasterize(...);

    // Member variables LAST
    std::vector<ProjectedGaussian> projected_;
};
```

**Order: public → private. Within each section: constructors → methods → data.**

### Rule of Zero / Rule of Five
- **Rule of Zero** (default): define no special member functions. The compiler-generated ones work because all your members manage themselves (`std::vector`, `std::unique_ptr`).
- **Rule of Five**: if you define one special member function (destructor, copy ctor, etc.), define all five.

Aim for Rule of Zero unless you're wrapping a raw resource (OpenGL handle, file descriptor).

### Always mark
- **`const` methods**: every method that doesn't modify the object. Aggressively. This is the single most important habit.
- **`override` on overridden virtuals**: catches signature mismatches.
- **`noexcept` on move ctor/move assign/destructor**: unlocks STL optimizations.
- **`[[nodiscard]]`** on functions returning values the caller shouldn't ignore.

### Constructors
- **Use member initializer lists**, not assignments in the body:
  ```cpp
  CpuRenderer(int w, int h) : width_(w), height_(h) {}  // ✓
  ```
- **Mark single-argument constructors `explicit`** unless implicit conversion is genuinely wanted.

### Virtual destructors
Any class with virtual methods needs a virtual destructor:
```cpp
class Renderer {
 public:
    virtual ~Renderer() = default;  // required for polymorphic deletion
    virtual void render(...) = 0;
};
```

---

## 6. Modern C++ Features

### Use these
- **`auto`** for obvious types: `auto it = vec.begin();`, `auto cloud = load_ply(...)`. Not for non-obvious types.
- **Range-based for**: `for (const auto& g : cloud) { ... }`.
- **Uniform init `{}`**: `Vec3 v{1, 2, 3};`. Prevents narrowing conversions.
- **`nullptr`**, never `NULL` or `0` for pointers.
- **`enum class`**, never plain `enum`.
- **Smart pointers**: `std::unique_ptr` by default. `std::shared_ptr` only when genuinely shared ownership.
- **`std::span` (C++20)**: better than `(T* ptr, size_t n)` argument pairs.
- **`std::string_view`** for read-only string parameters.
- **`constexpr`** for compile-time constants.
- **Structured bindings**: `auto [ok, value] = try_parse(...);`.
- **`[[nodiscard]]`**, `[[maybe_unused]]` attributes where helpful.

### Be careful with
- **`auto&` vs `auto`**: easy to accidentally copy when you wanted a reference. Be explicit when it matters.
- **`std::shared_ptr`**: overused. Default to `unique_ptr` + references.

### Avoid
- **Raw `new` / `delete`**: use `std::make_unique`, stack allocation.
- **C-style casts** `(int)x`: use `static_cast<int>(x)`. Searchable, loud.
- **C-style arrays**: use `std::array<T, N>` or `std::vector<T>`.
- **Macros for constants/functions**: use `constexpr` or `inline`.
- **`using namespace std;` in headers**: never.

---

## 7. Formatting

### `.clang-format` (project root)

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
DerivePointerAlignment: false
AllowShortFunctionsOnASingleLine: Empty
SpaceAfterTemplateKeyword: true
AlwaysBreakTemplateDeclarations: Yes
IncludeBlocks: Regroup
SortIncludes: CaseInsensitive
NamespaceIndentation: None
FixNamespaceComments: true
```

### Why these choices
- **Base: Google** — most well-tested style.
- **IndentWidth 4** — easier to read than Google's default 2.
- **ColumnLimit 100** — modern displays, modern style.
- **PointerAlignment Left** — `int* p`, not `int *p`. Type-centric.
- **IncludeBlocks Regroup** — clang-format will sort and group your includes per Section 3.
- **FixNamespaceComments** — auto-adds `// namespace foo` closing comments.

### Commit a `.clang-format` from day one
Enable format-on-save in your editor. After that, formatting is never discussed.

---

## 8. CMake

### Top-level `CMakeLists.txt` template

```cmake
cmake_minimum_required(VERSION 3.20)
project(gs-viewer
    VERSION 0.1.0
    LANGUAGES CXX
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies via FetchContent
include(FetchContent)

FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(Eigen glfw)

# Project sources
add_subdirectory(src)

# Tests
enable_testing()
```

### Modern CMake rules

1. **Always use `target_*` commands, never global ones.**
   - `target_include_directories(my_target PRIVATE ...)` ✓
   - `include_directories(...)` ✗
   - `target_link_libraries(my_target PRIVATE Eigen3::Eigen)` ✓
   - `link_libraries(...)` ✗

2. **Know PUBLIC / PRIVATE / INTERFACE.**
   - `PRIVATE` — only this target needs it. **Default.**
   - `PUBLIC` — this target AND anyone linking to it.
   - `INTERFACE` — only consumers (header-only libs).

3. **Out-of-source builds, always.** Build into `build/`, never the source tree.

4. **Use `FetchContent` for dependencies** unless you need vcpkg/Conan for some reason.

5. **Set `CMAKE_EXPORT_COMPILE_COMMANDS ON`** — generates `compile_commands.json` for clangd and other tools.

6. **Don't put binaries in `find_package` calls** — use targets like `Eigen3::Eigen`, not paths.

### `.gitignore` essentials
```
build/
.cache/
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
compile_commands.json
*.o
*.obj
```

---

## 9. Comments and Documentation

### Three levels of comments

**1. Per-file header comment**
```cpp
// EWA projection of 3D Gaussians to 2D screen-space ellipses.
// Implements Section 4 of Zwicker et al. 2001 (EWA Volume Splatting)
// and Section 4 of Kerbl et al. 2023 (3D Gaussian Splatting).
```

**2. Per-function (Doxygen-style for non-obvious functions)**
```cpp
/// Builds the 3D covariance matrix from a scale vector and unit quaternion.
/// See Kerbl 2023 Eq. 6: Σ = R S Sᵀ Rᵀ.
/// @param scale  Per-axis standard deviations (not log-space).
/// @param quat   Unit quaternion (w, x, y, z), assumed normalized.
Eigen::Matrix3f build_covariance_3d(const Eigen::Vector3f& scale,
                                    const Eigen::Quaternionf& quat);
```

**3. Inline (the "why," never the "what")**
```cpp
// Bad — repeats the code:
i += 1;  // increment i

// Good — explains the reason:
if (depth < kNearPlane) continue;  // avoid divide-by-zero in projection
```

### For this project specifically
**Comment the math.** When implementing a paper formula, cite the equation. `// Kerbl 2023, Eq. 4` is one line and saves you hours of future-you confusion.

### TODO conventions
```cpp
// TODO(yourname): brief description
// FIXME: what's broken and roughly why
// NOTE: explanation of a non-obvious choice
// HACK: temporary workaround, should be removed
```

---

## 10. VS Code Setup

### Essential extensions

| Extension | Purpose |
|---|---|
| **clangd** (LLVM) | Language server. Autocomplete, errors, navigation. **Critical.** |
| **CMake Tools** (Microsoft) | Configure/build/run CMake from the IDE. |
| **CMake** (twxs) | Syntax highlighting for CMakeLists.txt. |
| **CodeLLDB** | Debugger. Better than gdb for most things. |
| **GitLens** | Inline git blame, history. |
| **Better Comments** | Highlights TODO, FIXME, etc. |

**Important**: install clangd, **disable IntelliSense from the Microsoft C/C++ extension**. They conflict. Use one — clangd is better.

### `.vscode/settings.json` (commit to repo)

```json
{
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.associations": {
        "*.h": "cpp",
        "*.cuh": "cuda-cpp"
    },
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureOnOpen": true,
    "clangd.arguments": [
        "--compile-commands-dir=${workspaceFolder}/build",
        "--background-index",
        "--clang-tidy",
        "--header-insertion=never"
    ],
    "[cpp]": {
        "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd"
    },
    "[cmake]": {
        "editor.defaultFormatter": "ms-vscode.cmake-tools"
    }
}
```

### `.clang-tidy` (project root)
Static analysis. Drop this in and clangd will surface issues inline:

```yaml
Checks: >
  -*,
  bugprone-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic
WarningsAsErrors: ''
HeaderFilterRegex: '.*'
```

### After this setup
- Save a file → formatted automatically
- Type code → clangd suggests, finds errors as you type
- F5 → build + debug
- F7 → build only
- Ctrl+Shift+P → CMake commands (configure, select target, etc.)
- Hover variable → shows type, value during debugging
- Right-click → "Go to Definition", "Find References", "Rename Symbol"

You stop thinking about most of this stuff.

---

## 11. Quick Reference Card

```
TYPES         GaussianCloud, CpuRenderer    UpperCamelCase
FUNCTIONS     project_to_screen()           snake_case
VARIABLES     view_matrix                   snake_case
MEMBERS       cloud_                        trailing_underscore_
CONSTANTS     kMaxDepth                     kCamelCase
ENUMS         enum class Mode { Cpu, Gpu }  enum class, UpperCamelCase values
NAMESPACES    gs, gs::splat                 lowercase, short
FILES         cpu_renderer.h/cpp            snake_case

HEADER GUARDS    #pragma once  (every header)
INCLUDE STYLE    "splat/projection.h"  (prefixed, from src/)
NULL POINTER     nullptr  (never NULL, never 0)
CASTS            static_cast<>  (never C-style)
ARRAYS           std::array, std::vector  (never C arrays)
OWNERSHIP        std::unique_ptr  (default; shared_ptr rarely)
CONST            on every method that doesn't modify
OVERRIDE         on every overridden virtual
[[nodiscard]]    on return values caller mustn't ignore

NEVER  using namespace std;  in headers
NEVER  raw new/delete
NEVER  C-style casts
NEVER  leading underscores in your names
NEVER  put binary files in find_package; use targets
ALWAYS clang-format on save
ALWAYS Rule of Zero unless wrapping a resource
ALWAYS virtual destructor in polymorphic base
```

---

## 12. References

### Official guides (skim, don't read end-to-end)
- **C++ Core Guidelines** (Stroustrup + Sutter) — https://isocpp.github.io/CppCoreGuidelines/
  The closest thing to authoritative. Long. Search when you have a specific question.
- **Google C++ Style Guide** — https://google.github.io/styleguide/cppguide.html
  Opinionated, well-explained, widely used.
- **LLVM Coding Standards** — https://llvm.org/docs/CodingStandards.html
  Similar to Google, slightly different in places.

### CMake
- **Modern CMake** (Henry Schreiner) — https://cliutils.gitlab.io/modern-cmake/
  The canonical introduction to post-3.5 CMake.
- **CMake official docs** — https://cmake.org/cmake/help/latest/

### Tooling
- **clangd documentation** — https://clangd.llvm.org/
- **clang-format options** — https://clang.llvm.org/docs/ClangFormatStyleOptions.html
- **clang-tidy checks** — https://clang.llvm.org/extra/clang-tidy/checks/list.html

### Reference projects to study
- **fmtlib/fmt** — Clean modern C++ library layout.
- **nerfstudio/gsplat** — Most architecturally clean 3DGS codebase. C++/CUDA + Python.
- **graphdeco-inria/gaussian-splatting** — The original 3DGS reference.

---

## Final principle

**Tools enforce conventions you don't think about. Conventions you don't think about are conventions you actually follow.**

Set up `.clang-format`, `.clang-tidy`, format-on-save, clangd diagnostics — once. Then write code, don't argue with yourself about style, and trust the tools.