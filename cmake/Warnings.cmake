# Project warning policy. Strict but pragmatic.
add_library(av_warnings INTERFACE)

target_compile_options(av_warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Werror=return-type
    -Werror=switch
    # Quiet noisy third-party headers (Boost, OpenEXR, etc.):
    -Wno-deprecated-declarations
)

# Visibility: hide by default, export deliberately.
target_compile_options(av_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-fvisibility-inlines-hidden>
    -fvisibility=hidden
)

add_library(av::warnings ALIAS av_warnings)
