
# Find the finufft library (tomocam needs its GPU flavor, cufinufft)
#
# This module defines:
#   finufft_FOUND
#   finufft_INCLUDE_DIR   (module mode only)
#   finufft_LIBRARIES     (module mode only)
#
# Targets:
#   finufft::cufinufft
#
# Set finufft_DIR to the directory containing finufftConfig.cmake, e.g.:
#   cmake -Dfinufft_DIR=~/finufft/lib/cmake/finufft ..
#
# For module-mode fallback set finufft_ROOT to the installation prefix, or
# rely on the hardcoded ~/finufft search path.

include(FindPackageHandleStandardArgs)

# finufftConfig.cmake unconditionally calls find_dependency(OpenMP COMPONENTS C CXX),
# so the C language must be enabled before config mode is attempted, even though
# tomocam itself is C++/CUDA only.
enable_language(C)

# --- Config mode (preferred) ---
# finufft_DIR is the cmake config directory; try it first with NO_DEFAULT_PATH
# so the user's explicit choice wins, then fall through to the full cmake search.
find_package(finufft CONFIG QUIET
    PATHS
        "${finufft_DIR}"
        "$ENV{HOME}/finufft/lib/cmake/finufft"
        "$ENV{HOME}/finufft/lib64/cmake/finufft"
    NO_DEFAULT_PATH
)
if (NOT finufft_FOUND)
    find_package(finufft CONFIG QUIET)
endif()

if (finufft_FOUND)
    if (NOT TARGET finufft::cufinufft)
        message(FATAL_ERROR
            "finufft found in ${finufft_DIR} but it was built without CUDA "
            "support (no finufft::cufinufft target). Rebuild finufft with "
            "-DFINUFFT_USE_CUDA=ON."
        )
    endif()
    message(STATUS "Found finufft (config): ${finufft_DIR}")
    return()
endif()

# --- Module mode fallback ---
# ~/finufft is first so a local build takes priority over system installations.
# finufft_ROOT (CMake 3.12+) is honoured automatically by find_path/find_library.
set(finufft_SEARCH_PATHS
    ~/finufft
    /usr/local
    /opt/local
    /usr
    /opt
)

find_path(finufft_INCLUDE_DIR
    NAMES
        cufinufft.h
    HINTS
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        include
    PATHS
        ${finufft_SEARCH_PATHS}
)

find_library(finufft_LIBRARY
    NAMES
        cufinufft
    HINTS
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        lib64
        lib
    PATHS
        ${finufft_SEARCH_PATHS}
)

if (finufft_INCLUDE_DIR AND finufft_LIBRARY)
    set(finufft_FOUND TRUE)
    set(finufft_LIBRARIES ${finufft_LIBRARY})

    find_package_handle_standard_args(finufft
        REQUIRED_VARS
            finufft_INCLUDE_DIR
            finufft_LIBRARY
    )

    mark_as_advanced(
        finufft_INCLUDE_DIR
        finufft_LIBRARY
    )

    if (NOT TARGET finufft::cufinufft)
        add_library(finufft::cufinufft UNKNOWN IMPORTED GLOBAL)
        set_target_properties(finufft::cufinufft PROPERTIES IMPORTED_LOCATION ${finufft_LIBRARY})
        target_include_directories(finufft::cufinufft INTERFACE ${finufft_INCLUDE_DIR})
    endif()

    message(STATUS "Found finufft (module): ${finufft_LIBRARY}")
else()
    message(FATAL_ERROR
        "finufft (cufinufft) not found.\n"
        "  Config mode:  set finufft_DIR to the directory containing finufftConfig.cmake\n"
        "                e.g. -Dfinufft_DIR=~/finufft/lib/cmake/finufft\n"
        "  Module mode:  set finufft_ROOT to the installation prefix\n"
        "                e.g. -Dfinufft_ROOT=~/finufft"
    )
endif()
