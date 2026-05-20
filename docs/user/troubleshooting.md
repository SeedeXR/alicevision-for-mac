# Troubleshooting

Common runtime issues, with the specific fix and the source citation.

For **build-time** issues (CMake configure errors, missing Boost, metallib
failing to compile), see [Developer → Building from source](../dev/build.md).

## "xcrun metal: license has not been accepted"

```bash
sudo xcodebuild -license accept
```

The Metal toolchain refuses to compile until the Xcode license has been
accepted. Reproduces fresh-install on every machine.

## `default.metallib` not found at runtime

Every binary expects `default.metallib` next to itself
(`@executable_path/default.metallib`). The CMake rule
`av_install_metallib()` in `cmake/Metal.cmake` stages it there at install
time. If a binary errors with `Failed to load default.metallib`:

1. Confirm `build/src/shaders/default.metallib` exists (or
   `<prefix>/bin/default.metallib` for installed binaries).
2. Confirm it's alongside the binary (`ls $(dirname $(which aliceVision_cameraInit))`).
3. As a last resort, edit a tiny C++ test or use lldb to pass an absolute
   path: `MTL::Device::default_device().load_library("/absolute/path/default.metallib")`.

## `ALICEVISION_ROOT` not set / `config.ocio` warning

The 12 pipeline binaries expect a runtime data tree at
`<ALICEVISION_ROOT>/share/aliceVision/` containing `config.ocio`,
`cameraSensors.db`, and `luts/`. From `memory/mental_note.md` §7d:

> The runtime resource layout for `ALICEVISION_ROOT` is **NOT** a path to
> the source tree. It's a Unix-like install prefix:
> `<root>/share/aliceVision/` contains `config.ocio` + `luts/`.

Set it via:

```bash
export ALICEVISION_ROOT=/opt/alicevision-for-mac-0.1.0   # release tarball
# or, for a build tree:
export ALICEVISION_ROOT=$PWD/build/alicevision_root
```

If unset the binary falls back to an embedded path at startup and may work
for simple ops, but pipeline ops that re-resolve OCIO mid-run will crash.
Always set it for pipeline runs.

## Depth map is all `-2` (sentinels)

`-2` is the alpha-mask sentinel value. An all-`-2` view means **every pixel
was alpha-masked** — usually one of:

1. The prepareDenseScene EXR for that view is fully transparent (check
   `dense/<viewID>.exr` in an EXR viewer).
2. `sgmParams.maxSimilarity` is wrong — the S40 cascade in
   `memory/mental_note.md` §8i documents the canonical case where 99.9 % of
   voxels were rejected because the adapter forgot to scale `maxSimilarity`
   from `[0, 1]` to `[0, 254]`. If you're touching the adapter, audit each
   parameter against the upstream CUDA call site.
3. The view's T-camera list is empty (no co-visible cameras). Pre-S39 this
   would emit "0/N nearest cameras"; the SfM landmarks step must complete
   first.

## Pipeline hangs at "Retrieve best depth in volume"

Pre-S39 bug. Cause: `cuda_volumeRetrieveBestDepth` had a conditionally-
allocated output buffer (`if (_computeDepthSimMap) ...`) that upstream's
CUDA tolerated as a silent null-deref no-op but our Metal shim crashed on,
manifesting as a hang (the runtime SIGSEGV handler itself hung producing the
stack trace).

Fixed by defensive lazy-allocate in the adapter — see
`memory/mental_note.md` §8h. If you see this symptom on a different
forwarder, audit every adapter that takes a non-const
`CudaDeviceMemoryPitched&` for the same pattern.

## `aliceVision_meshing` fatals with "built without Alembic support"

You passed `--output foo.abc`. macOS doesn't link Alembic yet — use `.sfm`
(the dense SfMData JSON serializer) instead:

```bash
aliceVision_meshing -i ... --output dense.sfm --outputMesh mesh.obj
```

From `memory/mental_note.md` §8h-i: the SfM JSON carries the same
information the downstream `texturing` needs; PLY is an alternative.

## `aliceVision_texturing` writes only `texturedMesh.obj` + `.mtl`, no PNG

Default `--colorMappingFileType=none` skips the texture-atlas baking step.
Pass an explicit type to trigger baking:

```bash
aliceVision_texturing ... --colorMappingFileType png
```

Source: `memory/mental_note.md` §8h-ii.

## `texturing` reports "Found 0 image dimension(s)"

You fed `texturing` the raw `.ply` point cloud instead of the dense
SfMData. The `.sfm` file is the input that carries the views / intrinsics /
extrinsics for per-camera reprojection. Use the `--output` of `meshing` (the
`.sfm` file), not the `--outputMesh` (the `.ply`):

```bash
aliceVision_texturing -i dense.sfm --inputMesh mesh_filtered.obj ...
```

## Homebrew dylib version mismatch (`Library not loaded: ...`)

Symptom: binary runs `--help` fine but crashes mid-pipeline with
`dyld: Library not loaded: /opt/homebrew/.../<libname>.dylib`.

Two common causes:

1. **A Homebrew dep was upgraded** after the build (e.g. Boost bumped
   minor versions). The recorded RPATH is the install-time absolute
   path. Either rebuild, or `brew pin <package>` to lock the version.
2. **The release tarball is portable to a different machine** that lacks
   the dep. The tarball is *not* fully vendored — Homebrew runtime dylibs
   are required. Install them on the consumer machine: `brew install`
   the list from `Formula/alicevision-for-mac.rb`'s `depends_on` block.

## Slow first launch (Gatekeeper)

Standard `XprotectService` malware scan on first launch of any unsigned
Mach-O. Wait it out — subsequent launches are near-instant. If you
downloaded the tarball:

```bash
xattr -dr com.apple.quarantine /opt/alicevision-for-mac-0.1.0/
```

Until Phase 12 ships Developer-ID signed + `notarytool`-notarized binaries,
this is the expected workaround. Building from source via Homebrew avoids
Gatekeeper entirely (the formula builds locally).

## Won't run on Intel Mac

By design. `CMAKE_OSX_ARCHITECTURES = arm64` is forced; the build warns on
`x86_64`. There is no Rosetta path because the Metal kernels target
`apple-m1` and later.

## Filing a bug

The port is pre-release. Open an issue with:

```bash
sw_vers
sysctl -n machdep.cpu.brand_string
xcodebuild -version
xcrun metal --version
ctest --test-dir build --output-on-failure | tail -50
```

…plus the full output of the failing pipeline command.
