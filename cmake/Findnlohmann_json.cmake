
# Find the nlohmann_json library
#
# This module defines:
#   nlohmann_json_FOUND
#   nlohmann_json_INCLUDE_DIR
#
# Targets:
#   nlohmann_json::nlohmann_json
#
# Set nlohmann_json_DIR to the directory containing
# nlohmann_jsonConfig.cmake, e.g.:
#   cmake -Dnlohmann_json_DIR=~/json/lib/cmake/nlohmann_json ..
#
# For module-mode fallback set nlohmann_json_ROOT to the installation
# prefix, or rely on the hardcoded ~/json and ~/nlohmann_json search paths.
# Module mode does not check a requested version.

include(FindPackageHandleStandardArgs)

# --- Config mode (preferred) ---
# nlohmann_json ships its own version-aware CMake package config; try it
# first so `find_package(nlohmann_json <version>)` is honoured correctly.
find_package(nlohmann_json ${nlohmann_json_FIND_VERSION} CONFIG QUIET
    PATHS
        "${nlohmann_json_DIR}"
        "$ENV{HOME}/json/lib/cmake/nlohmann_json"
        "$ENV{HOME}/nlohmann_json/lib/cmake/nlohmann_json"
    NO_DEFAULT_PATH
)
if (NOT nlohmann_json_FOUND)
    find_package(nlohmann_json ${nlohmann_json_FIND_VERSION} CONFIG QUIET)
endif()

if (nlohmann_json_FOUND)
    message(STATUS "Found nlohmann_json (config): ${nlohmann_json_DIR}")
    return()
endif()

# --- Module mode fallback ---
set(nlohmann_json_SEARCH_PATHS
    ~/json
    ~/nlohmann_json
    /usr/local
    /opt/local
    /usr
    /opt
)

find_path(nlohmann_json_INCLUDE_DIR
    NAMES
        nlohmann/json.hpp
    HINTS
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        include
    PATHS
        ${nlohmann_json_SEARCH_PATHS}
)

find_package_handle_standard_args(nlohmann_json
    REQUIRED_VARS
        nlohmann_json_INCLUDE_DIR
)

if (nlohmann_json_FOUND)
    mark_as_advanced(nlohmann_json_INCLUDE_DIR)

    if (NOT TARGET nlohmann_json::nlohmann_json)
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
        target_include_directories(nlohmann_json::nlohmann_json INTERFACE ${nlohmann_json_INCLUDE_DIR})
    endif()

    message(STATUS "Found nlohmann_json (module): ${nlohmann_json_INCLUDE_DIR}")
else()
    message(FATAL_ERROR
        "nlohmann_json not found.\n"
        "  Config mode:  set nlohmann_json_DIR to the directory containing\n"
        "                nlohmann_jsonConfig.cmake, e.g.\n"
        "                -Dnlohmann_json_DIR=~/json/lib/cmake/nlohmann_json\n"
        "  Module mode:  set nlohmann_json_ROOT to the installation prefix\n"
        "                e.g. -Dnlohmann_json_ROOT=~/json\n"
        "                (module mode does not check the requested version)"
    )
endif()
