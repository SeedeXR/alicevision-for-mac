# Testing

The repository has three independent test suites:

| Suite | Driver | Where | Count |
|---|---|---|---|
| C++ kernel + adapter | `ctest` (CMake) | `tests/` | **37/37 pass** |
| Swift native UI | `swift test` | `meshroom-native/Tests/` | **115 pass** |
| Meshroom integration | manual smoke | `scripts/run_meshroom.sh` | Monstree mini3 end-to-end |

## C++ tests — `ctest`

```bash
cd build
ctest                                  # 37/37 expected
ctest -j8                              # parallelized; faster on M-series
ctest -j1                              # serialized; use if a test is flaky
ctest -R test_depth_pipeline --output-on-failure -V  # specific test, verbose
```

### `-j1` vs `-j8`

The full suite runs in ~23 s under `-j8` on M4 (S45 measurement). All 37
tests are independent — no shared filesystem state, no shared GPU
queue. Use `-j1` only if you suspect a flake (an actual GPU resource
contention issue) and want to isolate it. The known-flaky test list is
empty as of S45.

### Notable end-to-end tests

| Test | Validates |
|---|---|
| `test_metal_hello` | metal-cpp wiring + SAXPY/SIMD-reduction smoke. |
| `test_texture_smoke` | RAII textures, bilinear sampling, mipmap cascade. |
| `test_eig33` | Householder + QL on 3×3 random symmetric matrices vs Eigen reference. |
| `test_compute_normal` | PCA-plane-fit + cosine deviation on a tilted-plane scene. |
| `test_image_color_conversion` | sRGB → Lab × 2.55 against CPU reference. |
| `test_sgm_pipeline` | `init_sim → compute_similarity → optimize → retrieve_best_depth` on a synthetic scene. |
| `test_refine_pipeline` | `init_refine → refine_similarity → refine_best_depth`. |
| `test_depth_pipeline` | Full SGM → Bridge → Refine → Optimize chain (S24). |
| `test_multi_t_aggregation` | WTA + FP16 additive across multiple T cameras. |
| `test_device_mipmap_image` | `DeviceMipmapImage` end-to-end (upload + Lab + mipmap). |
| `test_device_cache` | `LRUCache` + `DeviceCache` eviction. |
| `test_device_stream_manager` | Multi-`MTLCommandQueue` parallel dispatch. |
| `test_volume_optimize_adaptive_p2` | S31 adaptive-P2 weighting path. |
| `test_upstream_adapter` | Adapter forwarder smoke (post-S38). |
| `test_sgm_pipeline_via_adapter` | SGM driven through the `cuda_*` adapter shim. |

### Regression workflow

After every kernel change:

1. `cmake --build build` — confirm clean compile.
2. `ctest -R test_<the_kernel>` — confirm the focused test still passes.
3. `ctest` — confirm 37/37 unaffected.
4. `cmake --build build --target aliceVision_depthMapEstimation` — confirm
   pipeline binary still links.
5. Optional: run a Monstree mini3 view via
   `aliceVision_depthMapEstimation --rangeSize 1` and eyeball the
   `_depthMap.exr` statistics (Min/Max/Avg). The S40 baseline is
   Min=-2, Max≈20-22, Avg≈3-4 on the SfM-reported depth range.

After any adapter change, **rerun the S41 audit** mentally — line up the
forwarder's parameter pre-processing against `dSV.cu` / `dDSM.cu`.

## Swift tests — `swift test`

```bash
cd meshroom-native
swift test                             # 115 tests expected to pass
```

The suite is split into two targets:

- **`ProjectModelTests`** — `.mg` round-trip and template-reference parser.
  Three fixtures in `Tests/ProjectModelTests/Fixtures/`:
  `appendTextAndFiles.mg` (real upstream fixture, 3 nodes, template form),
  `sharedTemplate.mg` (real upstream fixture, empty template),
  `photogrammetryMini.mg` (synthetic, exercises uid/outputs/parallel/groups).
- **`AppTests`** — covers `ProjectViewModel`, `GraphScheduler`,
  `GraphExecutor`, `GraphLayout`, `NodeUIDHasher`, `ConnectionEditing`,
  and the M5 chunked-execution path. Fixtures in
  `Tests/AppTests/Fixtures/`.

The Swift suite has **no external dependencies** — `swift test` builds
against the toolchain shipped with the active Xcode (Swift 5.9+ required for
the `platforms:` syntax in `Package.swift`).

## Meshroom integration

There is no automated test for the Python pipeline end-to-end; the smoke
test is the canonical Monstree mini3 run:

```bash
./scripts/run_meshroom.sh python meshroom-mac/bin/meshroom_batch \
    -i dataset_monstree/mini3 \
    -o /tmp/monstree-out \
    -p photogrammetryLegacy
```

Pass = the 11 pipeline stages all run to completion and produce a
`texturedMesh.obj` + `.mtl` + `.png` triple at the end. Fail = anything
short of that (most commonly the depth-map sentinels-only case from
[Troubleshooting](../user/troubleshooting.md)).

For the deeper smoke `scripts/phase12_install_smoke.sh` exercises the
install/package step.

## Code coverage

Not currently measured. The repo's invariant of "every kernel ships with a
matching test in `tests/`" is enforced socially via the
[Adding a kernel](adding-kernel.md) workflow, not by a coverage tool.

## CI

There is no CI today. Phase 12 includes setting up a GitHub Actions
workflow that runs ctest + swift test on a macos-14 (Apple Silicon)
runner. Tracking in `memory/todo.md`.
