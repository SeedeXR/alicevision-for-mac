#include "av/depth_map/Volume.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include "av/depth_map/adapter_profile.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace av::depth_map {

namespace {

// MSL-side param structs (must match the layout in volume_kernels.metal).
struct VolumeDimsGpu {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
};

struct VolumeInitUcharParams {
    VolumeDimsGpu dims;
    std::uint32_t value;
};

struct VolumeInitHalfParams {
    VolumeDimsGpu dims;
    std::uint32_t value_half_bits;
};

VolumeDimsGpu to_gpu(VolumeDims d)
{
    return { d.x, d.y, d.z };
}

// IEEE 754 binary32 → binary16 round-to-nearest-even (RNE).
//
// Reference implementation suitable for tests; not used in hot
// paths. Handles signed zero, normals, denormals, infinity, NaN.
// Lifted from common public-domain implementations (Mike Acton /
// Fabian Giesen "float_to_half_fast3_rtne" pattern, simplified).
std::uint16_t float_to_half_bits(float f)
{
    static_assert(sizeof(float) == 4, "float is not 4 bytes");
    const std::uint32_t fi = std::bit_cast<std::uint32_t>(f);

    const std::uint32_t sign = (fi >> 16) & 0x8000u;
    std::int32_t        exp  = static_cast<std::int32_t>((fi >> 23) & 0xffu) - 127 + 15;
    std::uint32_t       mant =  fi & 0x7fffffu;

    if (exp <= 0) {
        // Subnormal or underflow to zero.
        if (exp < -10) {
            // Too small to represent — round to ±0.
            return static_cast<std::uint16_t>(sign);
        }
        // Subnormal: pack the leading 1 + the shift.
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t round_bit = 1u << (shift - 1);
        std::uint32_t       half_mant = mant >> shift;
        if ((mant & ((1u << shift) - 1u)) > round_bit ||
            ((mant & ((1u << shift) - 1u)) == round_bit && (half_mant & 1u))) {
            half_mant += 1u;  // round-to-nearest-even
        }
        return static_cast<std::uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        // Overflow → infinity, or pass NaN through.
        if (((fi >> 23) & 0xffu) == 0xffu && mant != 0) {
            // NaN: keep some payload bits.
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    // Normal: pack with RNE on the discarded 13 bits.
    const std::uint32_t round_bit = 1u << 12;
    std::uint32_t       half_mant = mant >> 13;
    if ((mant & 0x1fffu) > round_bit ||
        ((mant & 0x1fffu) == round_bit && (half_mant & 1u))) {
        half_mant += 1u;
    }
    if (half_mant == 0x400u) {
        // Mantissa overflow after rounding bumps exponent.
        half_mant = 0;
        exp += 1;
        if (exp >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | half_mant);
}

}  // namespace

// S48: per-axis specialized PSO set for Volume::optimize. The 4 SGM-DP
// paths use only 2 distinct axis-swizzle configurations:
//   Y-axis paths (forward + reverse) → axis = {0, 1, 2}
//   X-axis paths (forward + reverse) → axis = {1, 0, 2}
// We build 2 PSO sets at Volume construction time and select by axis.x
// at dispatch time, eliminating per-thread runtime axis indirection.
struct OptimizeAxisPipelines {
    av::gpu::Pipeline init_y_slice_uchar;
    av::gpu::Pipeline get_xz_slice;
    av::gpu::Pipeline compute_best_z;
    av::gpu::Pipeline aggregate_cost;
};

static OptimizeAxisPipelines make_optimize_pipelines_fc(
    const av::gpu::Device& dev, int axis0, int axis1, int axis2)
{
    av::gpu::FunctionConstants fc;
    fc.set_int(0, axis0);
    fc.set_int(1, axis1);
    fc.set_int(2, axis2);
    return OptimizeAxisPipelines{
        dev.make_pipeline("av_volume_init_y_slice_uchar_fc",       fc),
        dev.make_pipeline("av_volume_get_xz_slice_uchar_to_uint_fc", fc),
        dev.make_pipeline("av_volume_compute_best_z_in_slice_fc",  fc),
        dev.make_pipeline("av_volume_aggregate_cost_at_x_fc",      fc),
    };
}

struct Volume::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline init_uchar;
    av::gpu::Pipeline init_half;
    av::gpu::Pipeline add_half;
    av::gpu::Pipeline update_uninit_uchar;
    av::gpu::Pipeline retrieve_best_depth;
    av::gpu::Pipeline compute_similarity;
    // S48: per-axis PSO sets for the SGM-DP sub-kernels.
    OptimizeAxisPipelines optimize_y;   // axis = {0, 1, 2}
    OptimizeAxisPipelines optimize_x;   // axis = {1, 0, 2}
    av::gpu::Pipeline refine_similarity;
    av::gpu::Pipeline refine_best_depth;

    Impl(av::gpu::Device& d,
         av::gpu::Pipeline p1,  av::gpu::Pipeline p2,
         av::gpu::Pipeline p3,  av::gpu::Pipeline p4,
         av::gpu::Pipeline p5,  av::gpu::Pipeline p6,
         OptimizeAxisPipelines oy, OptimizeAxisPipelines ox,
         av::gpu::Pipeline p11, av::gpu::Pipeline p12) noexcept
        : device(d),
          init_uchar          (std::move(p1)),
          init_half           (std::move(p2)),
          add_half            (std::move(p3)),
          update_uninit_uchar (std::move(p4)),
          retrieve_best_depth (std::move(p5)),
          compute_similarity  (std::move(p6)),
          optimize_y          (std::move(oy)),
          optimize_x          (std::move(ox)),
          refine_similarity   (std::move(p11)),
          refine_best_depth   (std::move(p12)) {}
};

Volume::Volume(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(
          dev,
          dev.make_pipeline("av_volume_init_uchar"),
          dev.make_pipeline("av_volume_init_half"),
          dev.make_pipeline("av_volume_add_half"),
          dev.make_pipeline("av_volume_update_uninitialized_uchar"),
          dev.make_pipeline("av_volume_retrieve_best_depth"),
          dev.make_pipeline("av_volume_compute_similarity"),
          make_optimize_pipelines_fc(dev, 0, 1, 2),  // Y-axis path
          make_optimize_pipelines_fc(dev, 1, 0, 2),  // X-axis path
          dev.make_pipeline("av_volume_refine_similarity"),
          dev.make_pipeline("av_volume_refine_best_depth")))
{}

Volume::Volume(Volume&&) noexcept            = default;
Volume& Volume::operator=(Volume&&) noexcept = default;
Volume::~Volume()                            = default;

void Volume::init_sim(av::gpu::Buffer& volume, VolumeDims dims, std::uint8_t value)
{
    using namespace av::gpu;
    const std::size_t need = dims.voxel_count() * sizeof(std::uint8_t);
    if (volume.size_bytes() < need)
        throw std::invalid_argument("Volume::init_sim: buffer too small");

    VolumeInitUcharParams p{ to_gpu(dims), static_cast<std::uint32_t>(value) };

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.init_sim")
      .set_pipeline(impl_->init_uchar)
      .set_buffer  (0, volume)
      .set_bytes   (1, &p, sizeof(p))
      .dispatch({ dims.x, dims.y, dims.z }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

void Volume::init_refine(av::gpu::Buffer& volume, VolumeDims dims, float value_float)
{
    using namespace av::gpu;
    const std::size_t need = dims.voxel_count() * sizeof(std::uint16_t);
    if (volume.size_bytes() < need)
        throw std::invalid_argument("Volume::init_refine: buffer too small");

    VolumeInitHalfParams p{
        to_gpu(dims),
        static_cast<std::uint32_t>(float_to_half_bits(value_float))
    };

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.init_refine")
      .set_pipeline(impl_->init_half)
      .set_buffer  (0, volume)
      .set_bytes   (1, &p, sizeof(p))
      .dispatch({ dims.x, dims.y, dims.z }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

void Volume::add_refine(av::gpu::Buffer& inout,
                        const av::gpu::Buffer& in,
                        VolumeDims dims)
{
    using namespace av::gpu;
    const std::size_t need = dims.voxel_count() * sizeof(std::uint16_t);
    if (inout.size_bytes() < need || in.size_bytes() < need)
        throw std::invalid_argument("Volume::add_refine: buffer too small");

    VolumeDimsGpu d = to_gpu(dims);

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.add_refine")
      .set_pipeline(impl_->add_half)
      .set_buffer  (0, inout)
      .set_buffer  (1, in)
      .set_bytes   (2, &d, sizeof(d))
      .dispatch({ dims.x, dims.y, dims.z }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

void Volume::update_uninitialized(av::gpu::Buffer& volume2nd,
                                   const av::gpu::Buffer& volume1st,
                                   VolumeDims dims)
{
    using namespace av::gpu;
    const std::size_t need = dims.voxel_count() * sizeof(std::uint8_t);
    if (volume2nd.size_bytes() < need || volume1st.size_bytes() < need)
        throw std::invalid_argument(
            "Volume::update_uninitialized: buffer too small");

    VolumeDimsGpu d = to_gpu(dims);

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.update_uninitialized")
      .set_pipeline(impl_->update_uninit_uchar)
      .set_buffer  (0, volume2nd)
      .set_buffer  (1, volume1st)
      .set_bytes   (2, &d, sizeof(d))
      .dispatch({ dims.x, dims.y, dims.z }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

namespace {

// MSL-side struct layout for RetrieveBestDepthParams. Must match
// the struct in volume_kernels.metal byte-for-byte.
struct RetrieveBestDepthParamsGpu {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
    std::uint32_t depthRangeBegin;
    std::uint32_t depthRangeEnd;
    std::uint32_t roiXBegin;
    std::uint32_t roiYBegin;
    std::int32_t  scaleStep;
    float         thicknessMultFactor;
    float         maxSimilarity;
};

}  // namespace

void Volume::retrieve_best_depth(av::gpu::Buffer&             out_depth_thickness,
                                 av::gpu::Buffer&             out_depth_sim,
                                 const av::gpu::Buffer&       in_depths,
                                 const av::gpu::Buffer&       volume,
                                 const DeviceCameraParams&    rc_camera,
                                 const RetrieveBestDepthParams& p)
{
    using namespace av::gpu;
    const std::size_t pixel_count   = std::size_t(p.dims.x) * std::size_t(p.dims.y);
    const std::size_t vol_bytes     = p.dims.voxel_count() * sizeof(std::uint8_t);
    const std::size_t depths_bytes  = std::size_t(p.dims.z) * sizeof(float);
    const std::size_t f2_out_bytes  = pixel_count * 2 * sizeof(float);
    if (out_depth_thickness.size_bytes() < f2_out_bytes ||
        out_depth_sim       .size_bytes() < f2_out_bytes)
        throw std::invalid_argument(
            "Volume::retrieve_best_depth: output buffer too small");
    if (in_depths.size_bytes() < depths_bytes)
        throw std::invalid_argument(
            "Volume::retrieve_best_depth: in_depths buffer too small");
    if (volume.size_bytes() < vol_bytes)
        throw std::invalid_argument(
            "Volume::retrieve_best_depth: volume buffer too small");

    RetrieveBestDepthParamsGpu params{};
    params.volDimX             = p.dims.x;
    params.volDimY             = p.dims.y;
    params.volDimZ             = p.dims.z;
    params.depthRangeBegin     = p.depth_range_begin;
    params.depthRangeEnd       = (p.depth_range_end == 0) ? p.dims.z : p.depth_range_end;
    params.roiXBegin           = p.roi_x_begin;
    params.roiYBegin           = p.roi_y_begin;
    params.scaleStep           = p.scale_step;
    params.thicknessMultFactor = p.thickness_mult_factor;
    params.maxSimilarity       = p.max_similarity;

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.retrieve_best_depth")
      .set_pipeline(impl_->retrieve_best_depth)
      .set_buffer  (0, out_depth_thickness)
      .set_buffer  (1, out_depth_sim)
      .set_buffer  (2, in_depths)
      .set_buffer  (3, volume)
      .set_bytes   (4, &rc_camera, sizeof(DeviceCameraParams))
      .set_bytes   (5, &params,    sizeof(params))
      .dispatch({ p.dims.x, p.dims.y, 1u }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

namespace {

// MSL-side struct layout for ComputeSimilarityParams.
struct ComputeSimilarityParamsGpu {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
    std::uint32_t rcSgmLevelWidth;
    std::uint32_t rcSgmLevelHeight;
    std::uint32_t tcSgmLevelWidth;
    std::uint32_t tcSgmLevelHeight;
    float         rcMipmapLevel;
    std::int32_t  stepXY;
    std::int32_t  wsh;
    float         invGammaC;
    float         invGammaP;
    std::uint32_t useConsistentScale;
    std::uint32_t depthRangeBegin;
    std::uint32_t depthRangeEnd;
    std::uint32_t roiXBegin;
    std::uint32_t roiYBegin;
    std::uint32_t roiWidth;
    std::uint32_t roiHeight;
};

}  // namespace

void Volume::compute_similarity(av::gpu::Buffer&             out_volume_1st,
                                av::gpu::Buffer&             out_volume_2nd,
                                const av::gpu::Buffer&       in_depths,
                                const av::gpu::Texture&      rc_mipmap,
                                const av::gpu::Texture&      tc_mipmap,
                                const DeviceCameraParams&    rc_camera,
                                const DeviceCameraParams&    tc_camera,
                                const ComputeSimilarityParams& p)
{
    using namespace av::gpu;
    const std::size_t vol_bytes    = p.dims.voxel_count() * sizeof(std::uint8_t);
    const std::size_t depths_bytes = std::size_t(p.dims.z) * sizeof(float);
    if (out_volume_1st.size_bytes() < vol_bytes ||
        out_volume_2nd.size_bytes() < vol_bytes)
        throw std::invalid_argument(
            "Volume::compute_similarity: output volume too small");
    if (in_depths.size_bytes() < depths_bytes)
        throw std::invalid_argument(
            "Volume::compute_similarity: in_depths too small");

    const std::uint32_t roiW = (p.roi_width  == 0) ? p.dims.x : p.roi_width;
    const std::uint32_t roiH = (p.roi_height == 0) ? p.dims.y : p.roi_height;
    const std::uint32_t zEnd = (p.depth_range_end == 0) ? p.dims.z : p.depth_range_end;
    if (zEnd > p.dims.z || p.depth_range_begin >= zEnd)
        throw std::invalid_argument(
            "Volume::compute_similarity: depth_range invalid");

    ComputeSimilarityParamsGpu params{};
    params.volDimX             = p.dims.x;
    params.volDimY             = p.dims.y;
    params.volDimZ             = p.dims.z;
    params.rcSgmLevelWidth     = p.rc_sgm_level_width;
    params.rcSgmLevelHeight    = p.rc_sgm_level_height;
    params.tcSgmLevelWidth     = p.tc_sgm_level_width;
    params.tcSgmLevelHeight    = p.tc_sgm_level_height;
    params.rcMipmapLevel       = p.rc_mipmap_level;
    params.stepXY              = p.step_xy;
    params.wsh                 = p.wsh;
    params.invGammaC           = p.inv_gamma_c;
    params.invGammaP           = p.inv_gamma_p;
    params.useConsistentScale  = p.use_consistent_scale;
    params.depthRangeBegin     = p.depth_range_begin;
    params.depthRangeEnd       = zEnd;
    params.roiXBegin           = p.roi_x_begin;
    params.roiYBegin           = p.roi_y_begin;
    params.roiWidth            = roiW;
    params.roiHeight           = roiH;

    const std::uint32_t zCount = zEnd - p.depth_range_begin;

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.compute_similarity")
      .set_pipeline(impl_->compute_similarity)
      .set_buffer  (0, out_volume_1st)
      .set_buffer  (1, out_volume_2nd)
      .set_buffer  (2, in_depths)
      .set_bytes   (3, &rc_camera, sizeof(DeviceCameraParams))
      .set_bytes   (4, &tc_camera, sizeof(DeviceCameraParams))
      .set_bytes   (5, &params,    sizeof(params))
      .set_texture (0, rc_mipmap)
      .set_texture (1, tc_mipmap)
      .dispatch({ roiW, roiH, zCount },
                { 4u, 2u, 8u });   // 64 threads, 3D (S45: Z-coherent texture sampling — 66% faster than the previous {16,4,1} on M4)
    cb.commit_and_wait();
}

// ============================================================
// optimize: SGM 4-direction DP aggregation
// ============================================================

namespace {

struct OptimizeAxisGpu { std::int32_t axis0, axis1, axis2; };
struct InitYSliceParamsGpu {
    std::uint32_t volDimX, volDimY, volDimZ;
    OptimizeAxisGpu axis;
    std::int32_t  y;
    std::uint32_t value;
};
struct GetXZSliceParamsGpu {
    std::uint32_t volDimX, volDimY, volDimZ;
    OptimizeAxisGpu axis;
    std::int32_t  y;
};
struct BestZParamsGpu {
    std::uint32_t volDimXa;
    std::uint32_t volDimZ;
};
struct AggregateParamsGpu {
    std::uint32_t volDimX, volDimY, volDimZ;
    OptimizeAxisGpu axis;
    std::int32_t  y;
    float         P1;
    float         P2_abs;
    std::uint32_t filteringIndex;
    // -------- adaptive-P2 fields (Session 22) --------
    std::uint32_t adaptive_p2;
    std::int32_t  ySign;
    std::int32_t  stepXY;
    std::int32_t  roiXBegin;
    std::int32_t  roiYBegin;
    std::uint32_t rcLevelWidth;
    std::uint32_t rcLevelHeight;
    float         rcMipmapLevel;
    float         p2_sig_mid;
};

}  // namespace

void Volume::optimize(av::gpu::Buffer&        out_volume,
                      av::gpu::Buffer&        slice_a,
                      av::gpu::Buffer&        slice_b,
                      av::gpu::Buffer&        axis_acc,
                      const av::gpu::Buffer&  in_volume,
                      const OptimizeParams&   p,
                      const av::gpu::Texture* rc_mipmap)
{
    using namespace av::gpu;
    const std::size_t vol_bytes = p.dims.voxel_count() * sizeof(std::uint8_t);
    if (in_volume .size_bytes() < vol_bytes ||
        out_volume.size_bytes() < vol_bytes)
        throw std::invalid_argument("Volume::optimize: volume buffer too small");

    const std::uint32_t lastZ = (p.last_depth_index == 0)
                                 ? p.dims.z : p.last_depth_index;
    if (lastZ > p.dims.z)
        throw std::invalid_argument("Volume::optimize: last_depth_index > dims.z");

    const std::uint32_t maxXY = std::max(p.dims.x, p.dims.y);
    const std::size_t slice_bytes =
        std::size_t(maxXY) * std::size_t(lastZ) * sizeof(std::uint32_t);
    const std::size_t axis_bytes = std::size_t(maxXY) * sizeof(std::uint32_t);
    if (slice_a.size_bytes() < slice_bytes ||
        slice_b.size_bytes() < slice_bytes ||
        axis_acc.size_bytes() < axis_bytes)
        throw std::invalid_argument(
            "Volume::optimize: scratch buffer too small");

    if (p.adaptive_p2 && rc_mipmap == nullptr)
        throw std::invalid_argument(
            "Volume::optimize: adaptive_p2 requires rc_mipmap texture");

    // The 4 paths: (axis, invY). axisT for X-axis path is (1, 0, 2)
    // and for Y-axis path is (0, 1, 2). Both directions per axis
    // (forward + reverse) → 4 paths total.
    struct Path { OptimizeAxisGpu axis; bool invY; };
    const Path paths[4] = {
        { { 0, 1, 2 }, false },   // Y forward
        { { 0, 1, 2 }, true  },   // Y reverse
        { { 1, 0, 2 }, false },   // X forward (axisT swaps X and Y)
        { { 1, 0, 2 }, true  },   // X reverse
    };

    // Volume dims (for kernels).
    const std::uint32_t volX = p.dims.x;
    const std::uint32_t volY = p.dims.y;
    const std::uint32_t volZ = p.dims.z;

    std::uint32_t filteringIndex = 0;
    for (const Path& path : paths) {
        AV_ADAPTER_PROFILE_SCOPE("vO_path_total");

        // Per-path dims in axisT-permuted coords.
        const std::uint32_t vol_dim_arr[3] = { volX, volY, volZ };
        const std::uint32_t axDimX = vol_dim_arr[path.axis.axis0];
        const std::uint32_t axDimY = vol_dim_arr[path.axis.axis1];
        const std::uint32_t axDimZ = vol_dim_arr[path.axis.axis2];

        // S48: pick the function-constant-specialized PSO set for this
        // path's axis swizzle. {0,1,2} → Y-axis variant; {1,0,2} →
        // X-axis variant. The PSOs are precompiled at Volume ctor.
        const OptimizeAxisPipelines& psos = (path.axis.axis0 == 0)
            ? impl_->optimize_y
            : impl_->optimize_x;

        const Buffer* slice_for_y   = &slice_a;
        const Buffer* slice_for_ym1 = &slice_b;

        // BATCHED: all dispatches for this path go through a single
        // CommandBuffer + single compute encoder. Metal's automatic
        // hazard tracking inserts the necessary memory barriers between
        // dependent dispatches (e.g., bestZ writes axis_acc, aggregate
        // reads it). One commit_and_wait per path (4 total per
        // optimize call) instead of ~3×axDimY commits.
        //
        // Why per-path and not per-tile: the ax-permuted dims change
        // between paths, so threadgroup dispatch shapes differ; also
        // keeping per-path keeps GPU work submission steady-state
        // without ballooning the command buffer size.
        CommandBuffer cb(impl_->device);
        cb.set_label("volume.optimize.path");

        // 1. Copy first XZ plane (Y=0) from in_volume into Ym1 slice.
        {
            const std::int32_t y0 = path.invY ? std::int32_t(axDimY) - 1 : 0;
            GetXZSliceParamsGpu gp{};
            gp.volDimX = volX; gp.volDimY = volY; gp.volDimZ = volZ;
            gp.axis = path.axis;
            gp.y = y0;
            cb.set_pipeline(psos.get_xz_slice)
              .set_buffer  (0, *const_cast<Buffer*>(slice_for_ym1))
              .set_buffer  (1, const_cast<Buffer&>(in_volume))
              .set_bytes   (2, &gp, sizeof(gp))
              .dispatch({ axDimX, axDimZ, 1u }, { 16u, 4u, 1u });
        }

        // 2. Set out_volume's Y=0 plane to 255.
        {
            InitYSliceParamsGpu ip{};
            ip.volDimX = volX; ip.volDimY = volY; ip.volDimZ = volZ;
            ip.axis = path.axis;
            ip.y = path.invY ? std::int32_t(axDimY) - 1 : 0;
            ip.value = 255;
            cb.set_pipeline(psos.init_y_slice_uchar)
              .set_buffer  (0, out_volume)
              .set_bytes   (1, &ip, sizeof(ip))
              .dispatch({ axDimX, axDimZ, 1u }, { 16u, 4u, 1u });
        }

        // 3. Loop over Y from 1 to axDimY-1 (forward) or
        //    axDimY-2 down to 0 (reverse).
        for (std::uint32_t iy = 1; iy < axDimY; ++iy) {
            const std::int32_t y = path.invY
                ? std::int32_t(axDimY) - 1 - std::int32_t(iy)
                : std::int32_t(iy);

            // 3a. Compute best Z in Ym1 column (into axis_acc).
            {
                BestZParamsGpu bp{ axDimX, axDimZ };
                cb.set_pipeline(psos.compute_best_z)
                  .set_buffer  (0, *const_cast<Buffer*>(slice_for_ym1))
                  .set_buffer  (1, axis_acc)
                  .set_bytes   (2, &bp, sizeof(bp))
                  .dispatch_1d (psos.compute_best_z, axDimX);
            }

            // 3b. Copy in_volume's Y=y plane into slice_for_y.
            {
                GetXZSliceParamsGpu gp{};
                gp.volDimX = volX; gp.volDimY = volY; gp.volDimZ = volZ;
                gp.axis = path.axis;
                gp.y = y;
                cb.set_pipeline(psos.get_xz_slice)
                  .set_buffer  (0, *const_cast<Buffer*>(slice_for_y))
                  .set_buffer  (1, const_cast<Buffer&>(in_volume))
                  .set_bytes   (2, &gp, sizeof(gp))
                  .dispatch({ axDimX, axDimZ, 1u }, { 16u, 4u, 1u });
            }

            // 3c. Aggregate DP step.
            {
                AggregateParamsGpu ap{};
                ap.volDimX = volX; ap.volDimY = volY; ap.volDimZ = volZ;
                ap.axis = path.axis;
                ap.y = y;
                ap.P1 = p.p1;
                ap.P2_abs = p.p2_abs;
                ap.filteringIndex = filteringIndex;
                ap.adaptive_p2    = p.adaptive_p2 ? 1u : 0u;
                ap.ySign          = path.invY ? -1 : 1;
                ap.stepXY         = p.step_xy;
                ap.roiXBegin      = p.roi_x_begin;
                ap.roiYBegin      = p.roi_y_begin;
                ap.rcLevelWidth   = (p.rc_level_width  == 0u) ? volX : p.rc_level_width;
                ap.rcLevelHeight  = (p.rc_level_height == 0u) ? volY : p.rc_level_height;
                ap.rcMipmapLevel  = p.rc_mipmap_level;
                ap.p2_sig_mid     = p.p2_sig_mid;
                cb.set_pipeline(psos.aggregate_cost)
                  .set_buffer  (0, *const_cast<Buffer*>(slice_for_y))
                  .set_buffer  (1, *const_cast<Buffer*>(slice_for_ym1))
                  .set_buffer  (2, axis_acc)
                  .set_buffer  (3, out_volume)
                  .set_bytes   (4, &ap, sizeof(ap));
                if (rc_mipmap != nullptr) {
                    cb.set_texture(0, *rc_mipmap);
                }
                cb.dispatch({ axDimX, axDimZ, 1u }, { 16u, 4u, 1u });
            }

            // 3d. Swap slice_for_y ↔ slice_for_ym1.
            std::swap(slice_for_y, slice_for_ym1);
        }

        cb.commit_and_wait();
        ++filteringIndex;
    }
}

// ============================================================
// refine_similarity (Refine pass on FP16 volume)
// ============================================================

namespace {

struct RefineSimilarityParamsGpu {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
    std::uint32_t rcRefineLevelWidth;
    std::uint32_t rcRefineLevelHeight;
    std::uint32_t tcRefineLevelWidth;
    std::uint32_t tcRefineLevelHeight;
    float         rcMipmapLevel;
    std::int32_t  stepXY;
    std::int32_t  wsh;
    float         invGammaC;
    float         invGammaP;
    std::uint32_t useConsistentScale;
    std::uint32_t depthRangeBegin;
    std::uint32_t depthRangeEnd;
    std::uint32_t roiXBegin;
    std::uint32_t roiYBegin;
    std::uint32_t roiWidth;
    std::uint32_t roiHeight;
};

}  // namespace

void Volume::refine_similarity(av::gpu::Buffer&             inout_vol_sim_half,
                               const av::gpu::Buffer&       in_sgm_depth_pix_size_map,
                               const av::gpu::Texture&      rc_mipmap,
                               const av::gpu::Texture&      tc_mipmap,
                               const DeviceCameraParams&    rc_camera,
                               const DeviceCameraParams&    tc_camera,
                               const RefineSimilarityParams& p)
{
    using namespace av::gpu;
    const std::size_t vol_bytes = p.dims.voxel_count() * sizeof(std::uint16_t);
    const std::size_t dp_bytes  = std::size_t(p.dims.x) * std::size_t(p.dims.y)
                                * 2 * sizeof(float);
    if (inout_vol_sim_half.size_bytes() < vol_bytes)
        throw std::invalid_argument(
            "Volume::refine_similarity: inout_vol_sim_half too small");
    if (in_sgm_depth_pix_size_map.size_bytes() < dp_bytes)
        throw std::invalid_argument(
            "Volume::refine_similarity: in_sgm_depth_pix_size_map too small");

    const std::uint32_t roiW = (p.roi_width  == 0) ? p.dims.x : p.roi_width;
    const std::uint32_t roiH = (p.roi_height == 0) ? p.dims.y : p.roi_height;
    const std::uint32_t zEnd = (p.depth_range_end == 0) ? p.dims.z : p.depth_range_end;
    if (zEnd > p.dims.z || p.depth_range_begin >= zEnd)
        throw std::invalid_argument(
            "Volume::refine_similarity: depth_range invalid");

    RefineSimilarityParamsGpu params{};
    params.volDimX             = p.dims.x;
    params.volDimY             = p.dims.y;
    params.volDimZ             = p.dims.z;
    params.rcRefineLevelWidth  = p.rc_refine_level_width;
    params.rcRefineLevelHeight = p.rc_refine_level_height;
    params.tcRefineLevelWidth  = p.tc_refine_level_width;
    params.tcRefineLevelHeight = p.tc_refine_level_height;
    params.rcMipmapLevel       = p.rc_mipmap_level;
    params.stepXY              = p.step_xy;
    params.wsh                 = p.wsh;
    params.invGammaC           = p.inv_gamma_c;
    params.invGammaP           = p.inv_gamma_p;
    params.useConsistentScale  = p.use_consistent_scale;
    params.depthRangeBegin     = p.depth_range_begin;
    params.depthRangeEnd       = zEnd;
    params.roiXBegin           = p.roi_x_begin;
    params.roiYBegin           = p.roi_y_begin;
    params.roiWidth            = roiW;
    params.roiHeight           = roiH;

    const std::uint32_t zCount = zEnd - p.depth_range_begin;

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.refine_similarity")
      .set_pipeline(impl_->refine_similarity)
      .set_buffer  (0, inout_vol_sim_half)
      .set_buffer  (1, const_cast<Buffer&>(in_sgm_depth_pix_size_map))
      .set_bytes   (2, &rc_camera, sizeof(DeviceCameraParams))
      .set_bytes   (3, &tc_camera, sizeof(DeviceCameraParams))
      .set_bytes   (4, &params,    sizeof(params))
      .set_texture (0, rc_mipmap)
      .set_texture (1, tc_mipmap)
      .dispatch({ roiW, roiH, zCount },
                { 8u, 2u, 4u });   // S46: Z-tile shape (64 threads,
                                   // Z=4 best for refine's 31-deep volume;
                                   // see memory/perf_optimization_s45.md
                                   // "Phase 14 R3").
    cb.commit_and_wait();
}

// ============================================================
// refine_best_depth (Refine pipeline exit)
// ============================================================

namespace {

struct RefineBestDepthParamsGpu {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
    std::int32_t  samplesPerPixSize;
    std::int32_t  halfNbSamples;
    std::int32_t  halfNbDepths;
    float         twoTimesSigmaPowerTwo;
    std::uint32_t roiWidth;
    std::uint32_t roiHeight;
};

}  // namespace

void Volume::refine_best_depth(av::gpu::Buffer&             out_refine_depth_sim_map,
                               const av::gpu::Buffer&       in_sgm_depth_pix_size_map,
                               const av::gpu::Buffer&       in_vol_sim_half,
                               const RefineBestDepthParams& p)
{
    using namespace av::gpu;
    const std::size_t pix    = std::size_t(p.dims.x) * std::size_t(p.dims.y);
    const std::size_t out_bytes = pix * 2 * sizeof(float);
    const std::size_t dp_bytes  = pix * 2 * sizeof(float);
    const std::size_t vol_bytes = p.dims.voxel_count() * sizeof(std::uint16_t);
    if (out_refine_depth_sim_map.size_bytes() < out_bytes)
        throw std::invalid_argument(
            "Volume::refine_best_depth: out buffer too small");
    if (in_sgm_depth_pix_size_map.size_bytes() < dp_bytes)
        throw std::invalid_argument(
            "Volume::refine_best_depth: sgm depth-pix-size map too small");
    if (in_vol_sim_half.size_bytes() < vol_bytes)
        throw std::invalid_argument(
            "Volume::refine_best_depth: vol_sim_half too small");

    const std::uint32_t roiW = (p.roi_width  == 0) ? p.dims.x : p.roi_width;
    const std::uint32_t roiH = (p.roi_height == 0) ? p.dims.y : p.roi_height;
    if (p.samples_per_pix_size <= 0 ||
        p.half_nb_samples      <  0 ||
        p.half_nb_depths       <  0 ||
        p.two_times_sigma_pow_two <= 0.0f) {
        throw std::invalid_argument(
            "Volume::refine_best_depth: invalid params");
    }

    RefineBestDepthParamsGpu params{};
    params.volDimX               = p.dims.x;
    params.volDimY               = p.dims.y;
    params.volDimZ               = p.dims.z;
    params.samplesPerPixSize     = p.samples_per_pix_size;
    params.halfNbSamples         = p.half_nb_samples;
    params.halfNbDepths          = p.half_nb_depths;
    params.twoTimesSigmaPowerTwo = p.two_times_sigma_pow_two;
    params.roiWidth              = roiW;
    params.roiHeight             = roiH;

    CommandBuffer cb(impl_->device);
    cb.set_label("volume.refine_best_depth")
      .set_pipeline(impl_->refine_best_depth)
      .set_buffer  (0, out_refine_depth_sim_map)
      .set_buffer  (1, const_cast<Buffer&>(in_sgm_depth_pix_size_map))
      .set_buffer  (2, const_cast<Buffer&>(in_vol_sim_half))
      .set_bytes   (3, &params, sizeof(params))
      .dispatch({ roiW, roiH, 1u }, { 32u, 4u, 1u });
    cb.commit_and_wait();
}

}  // namespace av::depth_map
