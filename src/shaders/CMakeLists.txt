# Compile the project's MSL sources into default.metallib.
#
# This is the single .metallib for the early-port phase. As the
# kernel surface grows (depthMap port etc.) we will split into
# per-module metallibs and load by name.

av_compile_metal_library(
    TARGET   av_shaders
    OUTPUT   av_default.metallib
    STANDARD metal3.0
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/depth_map
    SOURCES
        hello.metal
        depth_map/eig33.metal
        depth_map/matrix_kernels.metal
        depth_map/texture_smoke.metal
        depth_map/patch_kernels.metal
        depth_map/color_kernels.metal
        depth_map/simstat_kernels.metal
        depth_map/comp_ncc.metal
        depth_map/color_conversion.metal
        depth_map/gaussian_filter.metal
        depth_map/mipmap_array.metal
        depth_map/volume_kernels.metal
        depth_map/volume_compute_similarity.metal
        depth_map/volume_optimize.metal
        depth_map/volume_refine_similarity.metal
        depth_map/volume_refine_best_depth.metal
        depth_map/depth_sim_map.metal
)
