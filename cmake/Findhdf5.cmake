
# Find the HDF5 library
#
# This is a workaround for Perlmutter: the Cray "cray-hdf5" module's own
# CMake config pulls in vendor targets that don't resolve cleanly, so this
# module does a plain path/library search instead of CMake's own FindHDF5.
#
# This module defines:
#   hdf5_FOUND
#   hdf5_INCLUDE_DIR
#   hdf5_LIBRARY
#
# Targets:
#   hdf5::hdf5
#
# Set HDF5_ROOT to the installation prefix, e.g.:
#   cmake -DHDF5_ROOT=/path/to/hdf5 ..
# (NERSC sets $HDF5_ROOT via environment modules.)

include(FindPackageHandleStandardArgs)

set(hdf5_SEARCH_PATHS
    /usr
    /usr/local
    /opt
    /opt/local
    ${HDF5_ROOT}
    $ENV{HDF5_ROOT}
    ${CMAKE_PREFIX_PATH}
)

find_path(hdf5_INCLUDE_DIR
    NAMES
        hdf5.h
    PATH_SUFFIXES
        include
    PATHS
        ${hdf5_SEARCH_PATHS}
)

find_library(hdf5_LIBRARY
    NAMES
        hdf5
    PATH_SUFFIXES
        lib64
        lib
    PATHS
        ${hdf5_SEARCH_PATHS}
)

find_package_handle_standard_args(hdf5
    REQUIRED_VARS
        hdf5_LIBRARY
        hdf5_INCLUDE_DIR
)

if (hdf5_FOUND)
    mark_as_advanced(
        hdf5_INCLUDE_DIR
        hdf5_LIBRARY
    )

    if (NOT TARGET hdf5::hdf5)
        add_library(hdf5::hdf5 INTERFACE IMPORTED GLOBAL)
        target_include_directories(hdf5::hdf5 INTERFACE ${hdf5_INCLUDE_DIR})
        target_link_libraries(hdf5::hdf5 INTERFACE ${hdf5_LIBRARY})
    endif()

    message(STATUS "Found hdf5 (module): ${hdf5_LIBRARY}")
else()
    message(FATAL_ERROR
        "hdf5 not found.\n"
        "  Module mode:  set HDF5_ROOT to the installation prefix\n"
        "                e.g. -DHDF5_ROOT=/path/to/hdf5"
    )
endif()
