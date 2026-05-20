if(AV_BUILD_HELLO_METAL)
    add_executable(test_metal_hello test_metal_hello.cpp)
    target_link_libraries(test_metal_hello
        PRIVATE av::gpu av::warnings
    )
    av_install_metallib(
        FROM       av_shaders
        EXECUTABLE test_metal_hello
    )
    add_test(NAME test_metal_hello COMMAND test_metal_hello)

    add_executable(test_texture_smoke test_texture_smoke.cpp)
    target_link_libraries(test_texture_smoke
        PRIVATE av::gpu av::warnings
    )
    av_install_metallib(
        FROM       av_shaders
        EXECUTABLE test_texture_smoke
    )
    add_test(NAME test_texture_smoke COMMAND test_texture_smoke)
endif()

if(Eigen3_FOUND)
    get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

    function(_av_add_metal_test name)
        add_executable(${name} ${name}.cpp)
        target_link_libraries(${name}
            PRIVATE av::depth_map_metal Eigen3::Eigen av::warnings
        )
        # Eigen headers trip strict warnings; treat as system.
        if(_eigen_inc)
            target_include_directories(${name} SYSTEM PRIVATE ${_eigen_inc})
        endif()
        av_install_metallib(FROM av_shaders EXECUTABLE ${name})
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    _av_add_metal_test(test_eig33)
    _av_add_metal_test(test_matrix)
    _av_add_metal_test(test_patch)
    _av_add_metal_test(test_color)
    _av_add_metal_test(test_simstat)
    _av_add_metal_test(test_comp_ncc)
    _av_add_metal_test(test_comp_ncc_custom_pattern)
    _av_add_metal_test(test_image_color_conversion)
    _av_add_metal_test(test_gaussian_filter)
    _av_add_metal_test(test_gaussian_filter_volume)
    _av_add_metal_test(test_volume_easy)
    _av_add_metal_test(test_retrieve_best_depth)
    _av_add_metal_test(test_compute_similarity)
    _av_add_metal_test(test_volume_optimize)
    _av_add_metal_test(test_volume_optimize_adaptive_p2)
    _av_add_metal_test(test_sgm_pipeline)
    _av_add_metal_test(test_sgm_accuracy)
    _av_add_metal_test(test_refine_similarity)
    _av_add_metal_test(test_refine_best_depth)
    _av_add_metal_test(test_refine_pipeline)
    _av_add_metal_test(test_depth_sim_map_copy)
    _av_add_metal_test(test_normal_map_upscale)
    _av_add_metal_test(test_smooth_thickness)
    _av_add_metal_test(test_upscale_depth_pixsize)
    # S48: removed RUN_SERIAL. The S46 author's "Metal-shared-state"
    # hypothesis was incorrect — the actual cause was an OOB read in
    # the bilinear path (both kernel and CPU reference) at edge ROI
    # pixels where floor((roiX-0.5)*ratio) = -1. Heap-state variance
    # across parallel ctest processes is what made the OOB-read
    # garbage flake under -j8. Fixed by clamping xp/yp to >= 0 in
    # both kernel and CPU reference (see depth_sim_map.metal and
    # test_upscale_depth_pixsize.cpp). The test now runs cleanly in
    # parallel.
    _av_add_metal_test(test_compute_normal)
    _av_add_metal_test(test_optimize_depth_sim_map)
    _av_add_metal_test(test_depth_pipeline)
    _av_add_metal_test(test_multi_t_aggregation)
    _av_add_metal_test(test_device_mipmap_image)
    _av_add_metal_test(test_device_cache)
    _av_add_metal_test(test_device_stream_manager)
    _av_add_metal_test(test_cuda_memory_shim)
    _av_add_metal_test(test_upstream_adapter)
    _av_add_metal_test(test_sgm_pipeline_via_adapter)

    # Gated on the upstream depthMap dep tree because the shim
    # library it exercises (av::cuda_host_shim) needs
    # aliceVision_image / aliceVision_mvsUtils.
    if(AV_BUILD_UPSTREAM_DEPTHMAP AND TARGET av::cuda_host_shim)
        add_executable(test_cuda_host_shim test_cuda_host_shim.cpp)
        target_link_libraries(test_cuda_host_shim
            PRIVATE
                av::cuda_host_shim
                av::depth_map_metal
                Eigen3::Eigen
                av::warnings
        )
        if(_eigen_inc)
            target_include_directories(test_cuda_host_shim
                SYSTEM PRIVATE ${_eigen_inc})
        endif()
        av_install_metallib(FROM av_shaders EXECUTABLE test_cuda_host_shim)
        add_test(NAME test_cuda_host_shim COMMAND test_cuda_host_shim)
    endif()
endif()
