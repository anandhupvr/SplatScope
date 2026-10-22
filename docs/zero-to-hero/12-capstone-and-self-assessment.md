# 12 — Capstone, Self-Assessment & Interview Bank

> **Mental model:** *you've learned something when you can explain it, decide with it, and defend the decision against a better-informed sceptic.* The capstone is a design review of your own finished system.

---

## 1. Capstone deliverables

| # | Deliverable | Built in |
|---|---|---|
| 1 | SplatScope viewer: CPU oracle + CUDA forward pass, SH degree 3, readback/interop selection, debug views, hotkeys | Module 09 P0–P12 |
| 2 | Design brief + ≥ 8 ADRs | Module 01 |
| 3 | Diagram set in `docs/architecture/` (context, components, classes, ownership + lifetime bars, sequence, state, data flow, memory, timeline, deployment) | Module 02 |
| 4 | Test suite: unit, component (fakes), stage, golden, metamorphic, loader fuzz target; CI green | Module 10 |
| 5 | Performance baseline + experiment log + performance report | Module 08 |
| 6 | Two design studies from module 11 (DS-1 recommended, plus one other) | Module 11 |
| 7 | **The final design review document** (§2) | This module |

---

## 2. The final design review

Write `docs/design-review.md` as if presenting SplatScope to a staff engineer who will decide whether it's ready for a team to build on. Then answer the questions in §3 *in writing*, inside the document.

```
1. Summary — what it is, for whom, status (one paragraph + screenshot)
2. Requirements recap — priorities and quality-attribute scenarios, each marked MET / NOT MET with evidence
3. Architecture — C4 L1–L3 + the composite-pattern diagram (module 04 §10); one paragraph per pattern: why
4. Ownership & memory — ownership tree, lifetime bars, lifetime classes, VRAM and host-RAM budgets vs measured
5. GPU pipeline — data-flow diagram, invariants per stage, how each is tested
6. Performance — roofline position per stage, Amdahl breakdown, top 5 experiments (kept and rejected)
7. Quality — test pyramid with counts, oracles used, sanitizer status, CI
8. Risks & debt — known limitations, what breaks first at 10× (module 11 §1 table), what you'd do next
9. Decisions — ADR index, and which ones you'd now decide differently, and why
```

Section 9 matters most. Senior engineers are distinguished less by never being wrong than by **noticing and saying when they were**.

---

## 3. Questions a staff engineer would ask in your review

**Problem and requirements**
1. Which quality attribute did you sacrifice most, and was that the right call?
2. Which assumption from your design brief turned out wrong? How did you find out?

**Architecture**
3. Show me a dependency that points the wrong way, or prove there isn't one.
4. If I swapped CUDA for Vulkan compute, which files change? (A good answer: one adapter.)
5. Where did you consider a render graph, a plugin system or an event bus, and why didn't you build one?

**Ownership**
6. Walk me through every resource's release path when the user closes the window mid-load.
7. What prevents a stale `DeviceSpan` from being used after `ensure_capacity_discard` grows?
8. What's the most dangerous borrow in the codebase?

**GPU**
9. Which stage is memory-bound and which is compute-bound? Show the evidence.
10. What's your render kernel's occupancy, what limits it, and does it matter?
11. Where are the sync points, and why can't you remove the first one?

**Quality**
12. What bug would your tests *not* catch?
13. Which metamorphic test found a real bug?

**Scale**
14. What breaks first at a 10× bigger scene? At 100 users?
15. If this became a team project tomorrow, what would you document or enforce first?

---

## 4. Self-assessment rubric

Rate yourself honestly per competency. **Evidence** means something you can point to in the repo.

| Competency | Aware | Practitioner | Senior | Evidence |
|---|---|---|---|---|
| **Problem framing** | Can list requirements | Writes quality-attribute scenarios with numbers; non-goals | Prioritises conflicting attributes explicitly and revisits when measurements disagree | Design brief; ADR "revisit if" clauses |
| **Diagramming** | Draws boxes and arrows | Uses the right diagram per question; separate dependency and data-flow views | Diagrams drive design discussions; generated diagrams cross-check the build | `docs/architecture/`; CMake graph comparison |
| **Ownership** | Knows `unique_ptr` | Every resource has an owner in a destructor; owner/view/lease distinction | Designs lifetime classes; reasons about async GPU lifetimes and deferred release | Wrappers + static_asserts; sanitizer-clean runs |
| **Architecture** | Recognises layering | Applies hexagonal ports, pipes & filters, DOD deliberately | Chooses patterns per scale; can argue the rejected alternatives | ADR-006; fake adapters; CMake-enforced layers |
| **Design patterns** | Names a few patterns | Uses Strategy, Factory, PIMPL, NVI, Command, State where they fit | Avoids patterns that don't pay; spots anti-patterns in review | Pattern map (module 05 §15) |
| **Data-oriented design** | Knows AoS vs SoA | Chooses layouts per access pattern; hot/cold split | Predicts performance from bytes touched before measuring | DOD micro-experiment log |
| **GPU model** | Writes a working kernel | Reasons about warps, coalescing, divergence, occupancy | Explains every Nsight finding from first principles | Occupancy table; micro-benchmarks |
| **Optimisation** | Tries things | Measures; one change at a time; oracle check | Roofline + Amdahl-driven ladder; logs rejected experiments | Experiment log; perf report |
| **Testing** | Writes unit tests | Pyramid with oracles; sanitizers; CI | Metamorphic/property tests; fuzzing; tests as executable invariants | `tests/`; CI config; fuzz target |
| **Production** | Knows the words | Handles hostile input, OOM, sticky errors; packaging | Designs for scale with seams; SLOs; failure isolation | Design studies; loader limits; degraded state |

---

## 5. "Explain it in two minutes" drills

Say each one out loud, without notes, to a rubber duck or a friend. If you stumble, reread the module.

1. Why does 3DGS sort by `(tile, depth)` instead of per pixel? (gpu-forward-pass 04)
2. What does the prefix sum do in the pipeline, and why not atomics? (gpu-forward-pass 04 §2)
3. RAII, and how it interacts with exceptions. (03 §3)
4. Owner vs view vs lease, with one SplatScope example each. (03 §4)
5. Why `Window` is non-movable. (03 §5.3)
6. Hexagonal architecture, using SplatScope's ports. (04 §2)
7. Dependency direction vs data-flow direction. (02 §3.3)
8. Why SoA for preprocess but maybe AoS for the render gather. (06 §3.1)
9. Warps, divergence and the render kernel's early exit. (07 §5)
10. Occupancy: what limits it, and why 100% isn't the goal. (07 §6)
11. The roofline model applied to preprocess and render. (08 §3)
12. Deferred release and why streaming needs it. (11 §2.3)
13. The 2D covariance formula and the bug when J is transposed. (gpu-forward-pass 04 §1.4, 01 A5)
14. One-way vs two-way doors, with one example of each in SplatScope. (01 §4.2)

---

## 6. Interview question bank (by topic, with where to study)

**C++ and ownership:** RAII and exceptions · Rule of 0/3/5 · what `std::move` does · `noexcept` moves and `vector` · `unique_ptr` vs `shared_ptr` cost · wrapping C handles · `span` dangling · PIMPL's destructor gotcha · type erasure · CRTP · designing a move-only GPU buffer class → *modules 03, 05*

**Architecture and design:** layered vs hexagonal · dependency inversion · when an event bus hurts · render graphs · SOLID with real examples · ADRs · one-way doors · how to make a decision reversible → *modules 01, 04, 05*

**Performance and data:** AoS vs SoA · cache lines and false sharing · roofline · Amdahl · benchmarking methodology · why `chrono` around a kernel is wrong → *modules 06, 08*

**GPU and CUDA:** warps and SIMT · coalescing · bank conflicts · occupancy limiters · `__syncthreads` rules · sticky errors · pinned memory · streams and the default stream · interop · prefix sum and radix sort → *module 07, gpu-forward-pass 03–04*

**3D Gaussian Splatting specifics:** Σ = RSSᵀRᵀ and why · EWA projection and the Jacobian · the +0.3 dilation · conic and 3σ radius · tile binning and key packing · front-to-back compositing and early termination · SH evaluation and the `f_rest` order → *gpu-forward-pass 04*

**System design:** out-of-core rendering · remote rendering service · Python extension ownership · multi-threaded viewer · deploying across GPU architectures · hostile input → *module 11*

**Behavioural (with SplatScope stories):** a trade-off you made · a bug you found with a tool (initcheck on A2) · a decision you reversed · an optimisation that *didn't* work and what you learned → *your ADRs and experiment log*

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **ADR** | Architecture Decision Record: one decision, its context, options and consequences |
| **AoS / SoA / AoSoA** | Array of structures / structure of arrays / blocks of SoA |
| **Arithmetic intensity** | FLOPs per byte of DRAM traffic; the x-axis of the roofline |
| **Bank conflict** | Shared-memory accesses by one warp serialised because they hit the same bank |
| **C4 model** | Context, Containers, Components, Code: zoom levels for architecture diagrams |
| **Coalescing** | Combining a warp's memory accesses into few sectors |
| **Composition root** | The one place (`Viewer`/`main`) where concrete objects are created and wired |
| **Conic** | The inverse 2D covariance (A, B, C), evaluated per pixel |
| **Deferred release** | Freeing a GPU resource only after the GPU has finished every frame that might use it |
| **Dependency inversion** | Policy and detail both depend on an abstraction owned by the policy side |
| **EWA splatting** | Elliptical weighted average: projecting Gaussians through a local linearisation of the projection |
| **Invariant** | A condition that's always true at defined points in a program |
| **L** | Number of (Gaussian, tile) instances in a frame |
| **Lease** | A scoped, temporary access right (map/unmap, lock/unlock) represented by an RAII object |
| **Lifetime class** | A group of resources whose size depends on the same quantity and changes at the same time |
| **Metamorphic test** | A test that transforms the input in a way with a known effect on the output |
| **NVI** | Non-Virtual Interface: a public non-virtual method wrapping a private virtual one |
| **Occupancy** | Resident warps ÷ maximum warps per SM |
| **Oracle** | Any source of the expected answer in a test |
| **PIMPL** | Pointer to implementation: hides private members behind an opaque pointer |
| **Port / adapter** | An interface owned by the core / its technology-specific implementation |
| **RAII** | Resource Acquisition Is Initialisation: release in destructors |
| **Roofline** | Attainable performance = min(peak compute, intensity × bandwidth) |
| **SIMT** | Single Instruction, Multiple Threads: warp-level lockstep execution |
| **SLO** | Service Level Objective: a measurable target, such as p95 frame time |
| **Sticky error** | A CUDA error that corrupts the context; every later call fails |
| **Walking skeleton** | The thinnest end-to-end version of a system that exercises the architecture |

---

## 8. Further reading (well-known, widely cited)

| Topic | Reference |
|---|---|
| Software design | John Ousterhout, *A Philosophy of Software Design* (deep modules, complexity) |
| Architecture & quality attributes | Bass, Clements & Kazman, *Software Architecture in Practice* (quality-attribute scenarios) |
| Diagrams | Simon Brown, *The C4 model* (c4model.com) |
| Decisions | Michael Nygard, "Documenting Architecture Decisions" (2011 blog post, the origin of ADRs) |
| C++ ownership | *C++ Core Guidelines* (Stroustrup & Sutter), sections R (resources), F (function parameter passing), C (classes) |
| Data-oriented design | Mike Acton, "Data-Oriented Design and C++" (CppCon 2014); Richard Fabian, *Data-Oriented Design* |
| GPU programming | NVIDIA *CUDA C++ Programming Guide* and *CUDA C++ Best Practices Guide*; Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors* |
| Render graphs | Yuriy O'Donnell, "FrameGraph: Extensible Rendering Architecture in Frostbite" (GDC 2017) |
| 3D Gaussian Splatting | Kerbl, Kopanas, Leimkühler & Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023); Zwicker et al., "EWA Splatting" (2002) |
| Services & reliability | Martin Kleppmann, *Designing Data-Intensive Applications*; Google, *Site Reliability Engineering* (SLOs, error budgets) |
