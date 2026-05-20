# Metal kernels

Authoritative inventory of every `kernel void av_*` entry point shipped in
`default.metallib`. Generated from `src/shaders/depth_map/*.metal`.

**Total: 41 kernel entry points across 16 `.metal` files** (the
`ARCHITECTURE.md` document quotes "35 distinct kernel entry points" — that
count omits the four S31 adaptive-P2 `_fc` variants and the
`av_create_mipmapped_array_level` deferred kernel; the canonical inventory
is below.)

## Phase 5 — Utility / validation harnesses

Each of these has a matching CPU reference in `tests/test_<area>.cpp` and
exists to stress the shared `.h` headers used elsewhere.

| `.metal` file | Kernel | CUDA source of truth |
|---|---|---|
| `eig33.metal` | `av_eig33_decompose` | `upstream/.../eig33.cuh` (Householder + QL on 3×3 symmetric) |
| `matrix_kernels.metal` | `av_matrix_validate` | `upstream/.../patch.cuh` (matrix helpers) |
| `patch_kernels.metal` | `av_patch_validate` | `upstream/.../patch.cuh` |
| `color_kernels.metal` | `av_color_validate` | `upstream/.../color.cuh` |
| `simstat_kernels.metal` | `av_simstat_validate` | `upstream/.../simStat.cuh` |
| `comp_ncc.metal` | `av_compNCC_validate_no_filter`, `av_compNCC_validate_filter`, `av_compNCC_customPattern_no_filter`, `av_compNCC_customPattern_filter` | `upstream/.../patch.cuh` (`compNCCby3DptsYK` + S31 custom pattern variant) |
| `texture_smoke.metal` | `av_texture_sample` | (no upstream — av::gpu smoke) |

**Shared headers** (no kernels, included from `.metal` + host C++):
`operators.h`, `matrix.h`, `Patch.h`, `color.h`, `SimStat.h`,
`DevicePatchPattern.h`, `eig33.h`, `volume_helpers.h`.

## Phase 6 — Image processing

| `.metal` file | Kernel | CUDA source of truth |
|---|---|---|
| `color_conversion.metal` | `av_rgb2lab` | `upstream/.../imageProcessing/deviceColorConversion.cu` |
| `gaussian_filter.metal` | `av_downscale_with_gaussian_blur` | `upstream/.../imageProcessing/deviceGaussianFilter.cu` |
| `gaussian_filter.metal` | `av_median_filter_3` | same |
| `gaussian_filter.metal` | `av_gaussian_blur_volume_z` (S31) | same |
| `gaussian_filter.metal` | `av_gaussian_blur_volume_xyz` (S31) | same |
| `mipmap_array.metal` | `av_create_mipmapped_array_level` | `upstream/.../imageProcessing/deviceMipmappedArray.cu` (deferred — custom cascade not bit-matched; we use `MTLBlitCommandEncoder generateMipmapsForTexture:` instead. See PORTING_NOTES §7.) |

## Phase 7 — SGM / Refine / Optimize / DepthSimMap

### Volume — cost-volume primitives

`volume_kernels.metal`:

| Kernel | Notes |
|---|---|
| `av_volume_init_uchar` | TSim = uchar |
| `av_volume_init_half` | TSimRefine = `_Float16` |
| `av_volume_add_half` | promote-add-demote (avoids fp16-add precision issues) |
| `av_volume_update_uninitialized_uchar` | sec-best fill |
| `av_volume_retrieve_best_depth` | WTA + thickness + sim normalization |

### Volume — similarity

`volume_compute_similarity.metal`:

| Kernel | Threadgroup |
|---|---|
| `av_volume_compute_similarity` | `{4, 2, 8}` after S45 (Z-coherent; was `{16, 4, 1}` pre-S45). |

`volume_refine_similarity.metal`:

| Kernel | Notes |
|---|---|
| `av_volume_refine_similarity` | FP16 cost volume + `compNCCby3DptsYK<TInvertAndFilter=true>` |

### Volume — optimize (SGM-DP)

`volume_optimize.metal` — 8 kernels, 4 baseline + 4 `_fc` adaptive-P2
variants (S31, `function_constants`-style branch).

| Kernel | Variant |
|---|---|
| `av_volume_init_y_slice_uchar` | baseline |
| `av_volume_init_y_slice_uchar_fc` | adaptive-P2 |
| `av_volume_get_xz_slice_uchar_to_uint` | baseline |
| `av_volume_get_xz_slice_uchar_to_uint_fc` | adaptive-P2 |
| `av_volume_compute_best_z_in_slice` | baseline |
| `av_volume_compute_best_z_in_slice_fc` | adaptive-P2 |
| `av_volume_aggregate_cost_at_x` | baseline |
| `av_volume_aggregate_cost_at_x_fc` | adaptive-P2 |

All eight dispatched onto a single command buffer + encoder per SGM path
(S44 optimization), 4 paths total per tile.

### Volume — refine best depth

`volume_refine_best_depth.metal`:

| Kernel |
|---|
| `av_volume_refine_best_depth` |

### DepthSimMap — post-processing (9 kernels)

`depth_sim_map.metal`:

| Kernel | Role |
|---|---|
| `av_depth_sim_map_copy_depth_only` | strip sim channel |
| `av_map_upscale_float3` | normal-map upscale (SGM-resolution → Refine-resolution) |
| `av_depth_thickness_smooth_thickness` | smooth + clamp (sub-FP32-ULP drift via `clamp` intrinsic fusion, PORTING_NOTES §8) |
| `av_compute_sgm_upscaled_depth_pix_size_map_nearest` | bridge SGM → Refine (alpha-mask threshold at level 0 of rc mipmap) |
| `av_compute_sgm_upscaled_depth_pix_size_map_bilinear` | bridge SGM → Refine (bilinear variant) |
| `av_depth_sim_map_compute_normal` | PCA normal estimation (FP32 `Stat3d` accumulators per S22) |
| `av_optimize_var_l_of_lab_to_w` | LAB.x gradient at chosen mip level → weight |
| `av_optimize_get_opt_depth_map` | extract depth from optimization buffer |
| `av_optimize_depth_sim_map` | N-iter gradient-descent fusion of SGM rough + Refine fine (chained-sigmoid blend; relaxed budget under `-ffast-math`, PORTING_NOTES §5) |

## Cross-reference

| Kernel name pattern | Host driver class | Source of truth |
|---|---|---|
| `av_eig33_*` | `Eig33` | `eig33.cuh` |
| `av_matrix_*` | `MatrixOps` | matrix helpers in `patch.cuh` |
| `av_patch_*` | `PatchOps` | `patch.cuh` |
| `av_color_*` | `ColorOps` | `color.cuh` |
| `av_simstat_*` | `SimStatOps` | `simStat.cuh` |
| `av_compNCC_*` | `CompNCC` | `patch.cuh::compNCCby3DptsYK` |
| `av_rgb2lab`, `av_gaussian_*`, `av_median_*` | `ImageColorConversion`, `GaussianFilter` | `imageProcessing/device*.cu` |
| `av_volume_*` | `Volume` | `planeSweeping/deviceSimilarityVolume.cu` + `volumeOptimize.cu` |
| `av_depth_sim_map_*`, `av_optimize_*`, `av_compute_sgm_upscaled_*`, `av_map_upscale_*`, `av_depth_thickness_*` | `DepthSimMap` | `planeSweeping/deviceDepthSimilarityMap.cu` |
| `av_create_mipmapped_array_level` | (not yet wired; built-in `generate_mipmaps()` substitutes) | `imageProcessing/deviceMipmappedArray.cu` |
| `av_texture_sample` | (test smoke only) | n/a |

## How they get into `default.metallib`

```mermaid
flowchart LR
    SRC[*.metal] -- "xcrun metal" --> AIR[*.air]
    AIR -- "xcrun metallib" --> LIB[default.metallib]
    LIB -- "av_install_metallib()" --> TST[tests/default.metallib]
    LIB -- "av_install_metallib()" --> BIN[build/default.metallib<br/>(next to each binary)]
```

The CMake module `cmake/Metal.cmake` runs `xcrun metal` (one `.air` per
`.metal`) followed by `xcrun metallib` (link all `.air` files into one
`default.metallib`). The custom function `av_install_metallib(FROM
av_shaders EXECUTABLE <bin>)` then copies the result alongside every test
and pipeline binary so `@executable_path/default.metallib` resolves.

## Numerical agreement budgets

| Test | Worst |err| or rel | Budget | Reason |
|---|---|---|---|
| `test_eig33` | rel `2.71e-6` eigenvalue, `<10⁻⁶` eigenvector cos | `1e-5` | FP32 Householder + QL on random symmetric |
| `test_image_color_conversion` | `2.94e-5` ΔL, `2.01e-4` Δa, `8.89e-5` Δb | `0.03` | sRGB → Lab × 2.55 (host ref also × 2.55) |
| `test_compute_normal` | cos median `6.00e-6`, p99 `1.04e-5`, worst `1.81e-5` | `1e-3` cos | well-conditioned PCA, FP32 |
| `test_comp_ncc_custom_pattern` | worst no-filter `2.18e-2`, filter `5.24e-2` | `5e-2` no-filter, `8e-2` filter | sigmoid amplifies subpart drift |
| `test_optimize_depth_sim_map` | depth rel `1.19e-7`, sim `2.04e-4` | depth `1e-5`, sim `1e-3` | chained sigmoid + `-ffast-math` |
| `test_smooth_thickness` | rel `<1e-6` | `1e-6` | sub-FP32-ULP, `clamp` intrinsic fusion |

For the rationale behind each of the relaxed budgets see the corresponding
section of [`PORTING_NOTES.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/PORTING_NOTES.md).
