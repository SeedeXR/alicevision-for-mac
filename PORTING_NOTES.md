# PORTING_NOTES.md — alicevision-for-mac

The architectural CUDA → Metal decisions that shape this port. Each
section follows the same template:

1. **What CUDA did** — the upstream behaviour we're replacing.
2. **What we do instead** — the Apple Silicon implementation.
3. **Why** — the constraint or trade-off driving the choice.
4. **Measured impact** — numbers from the validation tests.

The session references (S3, S22, etc.) point at `memory/handover_session.md`
entries where the decision was made and validated.

See also [BUILD.md](BUILD.md) for prerequisites and
[ARCHITECTURE.md](ARCHITECTURE.md) for the code tour.

---

## 1. FP64 → FP32 everywhere (Apple GPUs have no FP64)

### What CUDA did

Upstream's `eig33.cuh`, `cuda_stat3d` PCA accumulator, several
`Patch.cuh` helpers, and `depthSimMapComputeNormal_kernel` all use
`double` for the inner math (covariance accumulation, Householder
reflection coefficients, QL iteration thresholds).

### What we do instead

Every MSL kernel is FP32. The host-side mirrors used by CPU reference
tests stay FP64 so we can quantify the drift.

Specific changes:

- `eig33.metal`: Householder + symmetric tridiagonal QL on 3 × 3
  ported with `double → float` precision change throughout.
- `Stat3d` struct + `av_stat3d_*` helpers (S22, lives in
  `depth_sim_map.metal`) — FP32 accumulators replacing upstream's FP64
  `cuda_stat3d`.

### Why

Apple GPUs (Apple7/8/9 families) do not implement `double` in hardware.
Software emulation exists for some compute intrinsics but is not
performant and is not exposed in MSL. Forcing FP64 would either fail
to compile or fall off the GPU entirely.

### Measured impact

- `test_eig33` (S3): 4 096 random symmetric matrices vs Eigen
  `SelfAdjointEigenSolver`. Worst eigenvalue relative error **2.71e-6**;
  worst eigenvector cosine deviation **< 10⁻⁶**.
- `test_compute_normal` (S22): 2 880 valid pixels on a 64 × 48 tilted-
  plane scene. Cosine deviation median **6.00e-6**, p99 **1.04e-5**,
  worst **1.81e-5** — vastly inside the 1e-3 budget because a flat
  plane has a well-conditioned PCA so FP32 vs FP64 covariance barely
  matters.

---

## 2. `cuda_stat3d` FP32 accumulators (vs upstream FP64)

### What CUDA did

`cuda_stat3d` (in `eig33.cuh`) accumulates Σx, Σx², Σxy, … into `double`
fields. Used by `depthSimMapComputeNormal_kernel` for the PCA plane
fit.

### What we do instead

Ported as MSL `Stat3d` struct + `av_stat3d_*` helpers in
`depth_sim_map.metal` (S22). Same algorithm; all accumulators are
`float`.

### Why

Same constraint as §1 (no FP64 on Apple GPUs), but called out
separately because `stat3d` was the deferred follow-up from S3 — the
original eig33 port skipped it pending a kernel that actually needed
the PCA.

### Measured impact

Folded into the `test_compute_normal` numbers (§1): worst cosine
deviation 1.81e-5 on 2 880 valid pixels. The well-conditioned PCA on
the test plane means the accumulator precision contributes a tiny
fraction of the total drift.

---

## 3. `bool isCircle` → `int isCircle` for struct-layout stability

### What CUDA did

Upstream's `DevicePatchPattern` (the customPatchPattern struct, 868 B
total) has a `bool isCircle` field. CUDA `bool` is 1 byte with implicit
3-byte trailing padding in struct layout.

### What we do instead

In both the MSL mirror (`src/shaders/depth_map/DevicePatchPattern.h`)
and the host mirror
(`src/depth_map_metal/include/av/depth_map/DevicePatchPattern.hpp`),
the field is `int32_t isCircle`. The predicate test `if (isCircle)`
behaves identically (any non-zero is true). Decision made in S31 by
agent A.

### Why

`bool` in MSL is implementation-defined size, and `bool` in CUDA
versus the host C++ ABI versus the MSL compiler is fragile across the
three compilers. An explicit `int32_t` is mechanically equivalent for
the 0/non-zero predicate and guarantees a 4-byte slot in both layouts.

### Measured impact

`DevicePatchPattern` is **byte-identical (868 B)** across MSL and the
host mirror, validated by `static_assert(sizeof(...))` on the host
side. `test_comp_ncc_custom_pattern` runs 64 cases × 3-subpart
pattern: no-filter worst |err| **2.18e-2** (S7 budget 5e-2); filter
worst |err| **5.24e-2** (sigmoid amplifies subpart drift — see §5).

---

## 4. `xyz2lab` returns Lab × 2.55 to match upstream's [0, 255] convention

### What CUDA did

Upstream's `xyz2lab` multiplies the final L/a/b values by **2.55** so
they fit a uchar [0, 255] convention. Classical Lab has L ∈ [0, 100]
and a, b ∈ [-128, 128]; upstream's CUDA produces L ∈ [0, ~255] and a,
b ∈ [~-256, 256] (can exceed for saturated colours).

### What we do instead

Faithfully preserved. `color.h`'s `xyz2lab` multiplies by 2.55. The
S26 test failure during `DeviceMipmapImage` development is the canonical
gotcha: a downstream consumer assumed classical Lab ranges and asserted
L ∈ [0, 100], saw L ∈ [157, 222] in practice, and lost 45 minutes of
debugging to a unit/scale mismatch.

Documented at every producer surface (`color.h`, `DeviceMipmapImage.hpp`,
the test header).

### Why

`compNCCby3DptsYK<TInvertAndFilter>` and the Yoon-Kweon weight `CostYK`
were calibrated by upstream against these scaled Lab values. Changing
them would change the NCC numbers in a way that diverges from the
CUDA reference — exactly the parity we're trying to preserve.

### Measured impact

`test_image_color_conversion` (S8): 65 536-pixel synthetic gradient.
Worst |ΔL| **2.94e-5**, |Δa| **2.01e-4**, |Δb| **8.89e-5**, |Δα|
exactly **0** — all 100× inside the 0.03 budget. Because the CPU
reference also applies × 2.55, the test only catches a mismatch if the
GPU and CPU sides disagree; both produce the scaled output.

---

## 5. Metal `-ffast-math` loosens `exp()` in chained sigmoids

### What CUDA did

CUDA's standard `exp()` is IEEE-compliant by default; chained
sigmoids like `1 / (1 + exp(...))` produce numerically stable results
to ~FP32 ULP scale.

### What we do instead

Our MSL is compiled with `-ffast-math` (the `cmake/Metal.cmake`
default). `exp()` becomes the fast-math variant. The chained sigmoid
inside `optimize_depth_sim_map` drifts noticeably under this regime —
not enough to break the algorithm, but enough to require a relaxed
agreement budget.

### Why

`-ffast-math` is the Metal-compiler default for performance, and the
chained sigmoid is precisely the kind of code path where the
relaxation manifests (`exp` of values approaching 0, denominator
addition, division). Recompiling without `-ffast-math` would penalise
every other kernel for one consumer.

### Measured impact

`test_optimize_depth_sim_map` (S23): 32 × 24 uniform-plane scene with
SGM=4.0, Refine=4.05, 5 iterations. **Depth agreement single-FP32-ULP**
(worst rel **1.19e-7**); **sim agreement 2.04e-4 rel** vs CPU
reference. First case where we needed a relaxed budget (1e-3 sim vs
1e-5 depth) specifically because of fast-math; previously every
budget was set by FP32 arithmetic itself.

A second instance of fast-math drift surfaced in S20 (`smooth_thickness`):
the Metal compiler fuses `std::fmax(std::fmin(...))` into a `clamp`
intrinsic with slightly different rounding; sub-ULP drift on 1 457
pixels. Same root cause; logged separately because the manifestation
was via the `clamp` intrinsic rather than `exp` (see §8).

---

## 6. MSL `access::read_write` on multi-mip textures needs explicit LOD

### What CUDA did

CUDA texture objects bound to a mipmapped array support implicit-level
read/write via `tex2D(ref, x, y)` plus a separately-sampled level. The
upstream `cuda_rgb2lab_kernel` runs directly on the multi-mip CUDA
array — read at level 0, write at level 0, no LOD parameter needed.

### What we do instead

`DeviceMipmapImage::fill` (S26) introduces a working-texture hop:

1. Allocate a single-mip working texture (same level-0 dimensions).
2. Upload the source image into it.
3. Run `av_rgb2lab` on the *working* texture.
4. Memcpy via shared-storage from the working texture into level 0 of
   the destination multi-mip texture.
5. `generate_mipmaps()` on the destination.

### Why

`texture2d<T, access::read_write>` in MSL, when bound to a multi-mip
texture, requires the **explicit-LOD** `read(coord, level)` /
`write(value, coord, level)` form. Our `av_rgb2lab` uses the
implicit form, which compiles and runs against a single-mip texture
but produces wrong values on a multi-mip texture. Three responses
considered:

- (a) add `level()` to every kernel — invasive, pollutes the public
  kernel API.
- (b) write a parallel `av_rgb2lab_level` variant — duplicated kernel.
- (c) use a single-mip working texture and copy the result.

Picked (c); the memcpy is essentially free on UMA.

### Measured impact

Zero numerical drift — the staging-buffer copy is bit-exact.
Performance cost: one host-side memcpy on shared storage per `fill()`
call. `test_device_mipmap_image` (S26) runs in 0.04 s on a 128 × 96
input building a 64 × 48 level-0 plus 3 mip levels.

---

## 7. Built-in `generate_mipmaps()` substitutes for the custom CUDA cascade

### What CUDA did

`deviceMipmappedArray.cu` (390 LOC, 6 kernels, Phase 6 remainder) is
upstream's custom mipmap cascade. It deviates from a standard 2 ×
downsampler in subtle ways for the rc image's specific use case.

### What we do instead

`av::gpu::Texture::generate_mipmaps()` uses Metal's built-in
`MTLBlitCommandEncoder generateMipmapsForTexture:`, which is the
standard 2 × spatial average per level.

### Why

The rc mipmap is consumed by `compute_sgm_upscaled_depth_pix_size_map`
(alpha-mask threshold at level 0) and `optimize_var_l_of_lab_to_w`
(4-tap LAB.x gradient at the requested level). Neither requires exact
upstream-numerics parity for correctness — both are sampled-and-thresholded
consumers where a 2 × average is functionally equivalent. The custom
MSL port to bit-match upstream is cosmetic and **deliberately deferred**
(`memory/todo.md` Phase 6, "deviceMipmappedArray.cu" is still
unchecked).

### Measured impact

`test_device_mipmap_image` (S26) builds a 4-level cascade; the test
asserts Lab ranges per level. No accuracy regression measurable in any
downstream consumer. The S24 end-to-end test (full SGM → Refine →
Optimize on a fused 128 × 96 image) achieves 94.0 % lock-on within
1.5 SGM depth-plane steps — a number that depends on the rc mipmap
indirectly via the alpha mask, but is dominated by the SGM quantisation
step, not the mip-level numerics.

---

## 8. `-ffast-math` and `clamp` intrinsic fusion (smooth-thickness sub-ULP drift)

### What CUDA did

`depthThicknessMapSmoothThickness_kernel` clamps a value to
`[min_t, max_t]` via separate `fmin` / `fmax` calls. CUDA's
`std::fmax(std::fmin(d, max_t), min_t)` is a sequence of two
IEEE-rounded ops.

### What we do instead

Same source-level form in MSL. But the Metal compiler under
`-ffast-math` recognises the pattern and fuses it into the `clamp`
intrinsic, which has slightly different rounding semantics on the
boundary values.

### Why

Same root cause as §5: `-ffast-math` is the global default for the
metallib. Disabling it kernel-by-kernel is possible via a function
attribute but not currently in use; the drift is well below the
algorithmic tolerance of every consumer.

### Measured impact

`test_smooth_thickness` (S20): 1 457 pixels on the thickness-map
smoothing kernel. Agreement vs CPU FP32 reference within **1e-6
relative** — sub-FP32-ULP drift. Documented in the S20 handover entry
as "first case where the Metal compiler's clamp-intrinsic fusion
produced sub-ULP drift vs `std::fmax(std::fmin(...))`."

---

## 9. Coin-OR + LEMON dependencies stubbed (linearProgramming is INTERFACE-only)

### What CUDA did

Upstream's `multiview` library links against COIN-OR (`Coin::Clp`,
`Coin::CoinUtils`, `Coin::Osi`) for LP-based rotation averaging, and
against the COIN-OR LEMON graph library for graph algorithms in the
same module. The CMake configure path is `find_package(Clp) ...`.

### What we do instead

Stubbed in S30 (`memory/todo.md` Phase 2 Path C):

- **Coin::Clp / Coin::CoinUtils / Coin::Osi**: declared as empty
  `INTERFACE IMPORTED` targets in our root `CMakeLists.txt`. The
  upstream `linearProgramming` module is INTERFACE-only (header
  forwarder, no `.cpp` files), so no symbols are referenced at link
  time. depthMap's runtime does not solve any LP problem.
- **LEMON**: switched off via `add_compile_definitions(
  ALICEVISION_ROTATION_AVERAGING_WITH_BOOST)`. Upstream provides this
  flag explicitly so `multiview/rotationAveraging/l1.cpp` uses
  Boost.Graph in place of LEMON for the same routines. Homebrew's
  `lemon` package is the parser generator, not the COIN-OR graph
  library, so we couldn't link against it anyway.

### Why

Coin-OR is not on Homebrew. Building it from source (Path A in
`memory/todo.md`) costs hours of work for a feature depthMap doesn't
use. Stubbing is honest: any code that calls a Coin-OR function at
runtime would produce a clean link error, and we'd know to either port
that call site or build Coin-OR from source. So far, no such error has
appeared.

### Measured impact

`AV_BUILD_UPSTREAM_DEPTHMAP=ON` produces 10 `.a` files for the depthMap
dependency tree (`aliceVision_camera.a`, `aliceVision_geometry.a`,
`aliceVision_multiview.a`, `aliceVision_mvsData.a`,
`aliceVision_mvsUtils.a`, etc.) plus the 2 INTERFACE-only modules
(`stl`, `linearProgramming`). Zero link errors. Our 33 Metal-side
tests continue to pass (`ctest`: 33/33).

---

## 10. CMake-time patch of `l1.cpp` (clang 21 enforces [dcl.init])

### What CUDA did

`multiview/rotationAveraging/l1.cpp` has three sites of
`const Eigen::Matrix x;` (default-initialized const variables), then
immediately uses `(double*)x.data()` to cast-mutate the underlying
storage. Pre-clang-21 toolchains accept this; the cast-mutation pattern
is a known upstream idiom for "I want stack storage, not heap, and I
know I'll modify it through `.data()`".

### What we do instead

The root `CMakeLists.txt` does a configure-time `file(READ ... )` of
`upstream/src/aliceVision/multiview/rotationAveraging/l1.cpp`, then
two `string(REPLACE ...)` calls that drop the `const`:

```cmake
string(REPLACE "const aliceVision::Vec3 erij;"
               "aliceVision::Vec3 erij;" ...)
string(REPLACE "const Mat3 eRi;"
               "Mat3 eRi;" ...)
```

The patched copy is `file(WRITE ...)`-emitted to
`${CMAKE_BINARY_DIR}/upstream-patched/multiview/rotationAveraging/l1.cpp`.
After `add_subdirectory(upstream/.../multiview)` runs, we
`get_target_property(_mv_srcs aliceVision_multiview SOURCES)`,
`list(TRANSFORM REPLACE)` to swap the original path for the patched
path, and `set_target_properties(... SOURCES ...)` to apply.

`target_include_directories(aliceVision_multiview PRIVATE ...
rotationAveraging)` is also added so the patched file (which lives in
`build/`, not in `upstream/`) still finds its sibling `l1.hpp`.

### Why

Clang 21 (Apple's bundled compiler in Xcode 26) enforces the
[dcl.init] rule that const-qualified objects of class type without a
user-provided default constructor require an initializer. Eigen::Matrix
has no user-provided default constructor (it's defaulted/generated),
so `const Eigen::Matrix x;` is now ill-formed. The rule has been in
the standard since C++11 but enforcement tightened in clang 21; the
upstream code follows with `(double*)x.data()` cast-mutation anyway,
so dropping `const` changes nothing at runtime.

Per-target `CXX_STANDARD 17` does not help — the rule predates C++20.

The repository invariant is that `upstream/` is read-only reference.
Editing the file in place would violate that. The CMake-time patch is
the principled compromise: the upstream file is untouched on disk, the
patched copy lives in the build dir, and a clean rebuild always
regenerates it from upstream source.

### Measured impact

`aliceVision_multiview` builds successfully under
`AV_BUILD_UPSTREAM_DEPTHMAP=ON`. Our 33 existing tests are unaffected
because upstream code is build-only at this point — not linked into any
executable.

---

Related: [BUILD.md](BUILD.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
`memory/handover_session.md` · `memory/philosophy.md`
