# third-party vendored bits. metal-cpp is consumed as an interface
# include directory (see cmake/Metal.cmake).
#
# LEMON (S39): graph library used by upstream's `track` and
# `multiview/rotationAveraging/l1.cpp`. Homebrew's `lemon` package is the
# SQLite parser-generator (unrelated). The COIN-OR LEMON library is not
# packaged on Homebrew, so we vendor v1.3.1 here.
#
# We build only the non-LP/MIP portion: `base.cc`, `random.cc`, `color.cc`,
# `arg_parser.cc`. The LP/MIP backends (glpk, cplex, clp, cbc, soplex) and
# their .cc files are excluded — upstream code paths we use only depend on
# `list_graph.h`, `unionfind.h`, `bin_heap.h`, `core.h` and friends, all
# header-only.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lemon/lemon/list_graph.h")
    # Generate lemon/config.h with LP/MIP disabled.
    set(LEMON_VERSION 1.3.1)
    set(LEMON_HAVE_LONG_LONG 1)
    set(LEMON_USE_PTHREAD 1)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/lemon/lemon/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/lemon-generated/lemon/config.h"
        @ONLY
    )

    add_library(av_lemon STATIC
        lemon/lemon/base.cc
        lemon/lemon/color.cc
        lemon/lemon/random.cc
        lemon/lemon/arg_parser.cc
    )
    target_include_directories(av_lemon PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/lemon"
        "${CMAKE_CURRENT_BINARY_DIR}/lemon-generated"
    )
    set_target_properties(av_lemon PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
    # LEMON 1.3.1 has its own legacy warnings we don't care about.
    target_compile_options(av_lemon PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-shadow
        -Wno-sign-compare
        -Wno-mismatched-tags
    )
    # Provide upstream's expected target name as an alias so existing
    # link lines (`${LEMON_LIBRARY}` set to "av_lemon", or direct
    # references) work.
    add_library(lemon::lemon ALIAS av_lemon)
endif()
