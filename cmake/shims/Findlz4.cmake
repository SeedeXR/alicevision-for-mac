# Findlz4.cmake — module-mode shim for Homebrew's lz4 install.
#
# Homebrew ships lz4 with pkg-config metadata but no CMake config
# files, so upstream's `find_package(lz4 REQUIRED)` fails. This
# module locates the Homebrew install via pkg-config (when
# available) or by direct find_path/find_library, and exposes the
# canonical imported target `lz4::lz4` that upstream consumes.

if(TARGET lz4::lz4)
    set(lz4_FOUND TRUE)
    return()
endif()

find_path(LZ4_INCLUDE_DIR
    NAMES   lz4.h
    HINTS   ENV LZ4_ROOT
    PATHS   /opt/homebrew/include /usr/local/include /usr/include
)

find_library(LZ4_LIBRARY
    NAMES   lz4 liblz4
    HINTS   ENV LZ4_ROOT
    PATHS   /opt/homebrew/lib /usr/local/lib /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lz4
    REQUIRED_VARS LZ4_LIBRARY LZ4_INCLUDE_DIR
)

if(lz4_FOUND)
    add_library(lz4::lz4 UNKNOWN IMPORTED)
    set_target_properties(lz4::lz4 PROPERTIES
        IMPORTED_LOCATION             "${LZ4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
    )
    mark_as_advanced(LZ4_INCLUDE_DIR LZ4_LIBRARY)
endif()
