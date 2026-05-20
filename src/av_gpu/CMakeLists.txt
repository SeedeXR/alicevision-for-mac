# av_gpu — minimal Metal-cpp abstraction for AliceVision.
#
# This is a *skeleton* in the literal sense: we are intentionally not
# designing a generalized API up-front (see memory/philosophy.md §4).
# It grows by extraction from the first kernel ports. For now it ships
# just enough to spin up a device, load a metallib, build a compute
# pipeline, allocate Shared buffers, and dispatch a kernel.

add_library(av_gpu STATIC
    include/av/gpu/Device.hpp
    include/av/gpu/Buffer.hpp
    include/av/gpu/Texture.hpp
    include/av/gpu/Pipeline.hpp
    include/av/gpu/CommandBuffer.hpp
    include/av/gpu/Queue.hpp
    include/av/gpu/Errors.hpp
    src/Device.cpp
    src/Buffer.cpp
    src/Texture.cpp
    src/Pipeline.cpp
    src/CommandBuffer.cpp
    src/Queue.cpp
    src/metal_cpp_impl.cpp
)

target_include_directories(av_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(av_gpu
    PUBLIC  av::metal
    PRIVATE av::warnings
)

# We want strict warnings on our own code, but metal-cpp triggers
# -Wshadow / -Wold-style-cast in headers we don't own. Treat them as
# system includes (already done in Metal.cmake) and locally relax a
# few that even SYSTEM-include can't fully silence on clang 21.
target_compile_options(av_gpu PRIVATE
    -Wno-old-style-cast      # metal-cpp uses C-style casts internally
    -Wno-shadow              # metal-cpp shadows in inline accessors
)

set_target_properties(av_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(av::gpu ALIAS av_gpu)
