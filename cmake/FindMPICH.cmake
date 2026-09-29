
# Find the MPICH library
#
# This is a workaround for Perlmutter: the Cray "cray-mpich" module doesn't
# play well with CMake's built-in FindMPI (which can also pick up an unrelated
# MPI, e.g. a conda Open MPI), so this module does a plain path/library search.
#
# The cray-mpich module sets CRAY_MPICH_ROOTDIR (e.g. /opt/cray/pe/mpich/9.1.0);
# the compiler-specific install lives under <root>/ofi/<vendor>/<version>.
# That location is searched first and exclusively; the generic paths below are
# only used when the variable is not set.
#
# This module defines:
#   MPICH_FOUND
#   MPI_INCLUDE_DIR
#   MPI_LIBRARY
#
# Targets:
#   MPI::MPI
#
# Otherwise set MPI_ROOT to the installation prefix, e.g.:
#   cmake -DMPI_ROOT=/path/to/mpich ..

include(FindPackageHandleStandardArgs)

enable_language(C)

set(_mpich_hints)
if (DEFINED ENV{CRAY_MPICH_ROOTDIR} AND NOT "$ENV{CRAY_MPICH_ROOTDIR}" STREQUAL "")
    # vendor directory that matches the compiler, then any other
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(_vendors gnu)
    elseif (CMAKE_C_COMPILER_ID STREQUAL "NVHPC")
        set(_vendors nvidia)
    else()
        set(_vendors)
    endif()
    file(GLOB _installs "$ENV{CRAY_MPICH_ROOTDIR}/ofi/*/*")
    foreach(_v IN LISTS _vendors)
        file(GLOB _preferred "$ENV{CRAY_MPICH_ROOTDIR}/ofi/${_v}/*")
        list(PREPEND _installs ${_preferred})
    endforeach()
    list(REMOVE_DUPLICATES _installs)
    set(_mpich_hints ${_installs})
    set(_mpich_no_default NO_DEFAULT_PATH)
    message(STATUS "MPICH: using CRAY_MPICH_ROOTDIR=$ENV{CRAY_MPICH_ROOTDIR}")
else()
    set(_mpich_hints
        ${MPI_ROOT}
        $ENV{MPI_ROOT}
        $ENV{MPICH_DIR}
        /usr
        /usr/local
        /opt
        /opt/local
        ${CMAKE_PREFIX_PATH}
    )
    set(_mpich_no_default)
endif()

find_path(MPI_INCLUDE_DIR
    NAMES mpi.h
    PATH_SUFFIXES include
    PATHS ${_mpich_hints}
    ${_mpich_no_default}
)

# On Perlmutter libmpich.so is a symlink to libmpi_gnu.so (the runtime that
# srun and mpi4py use)
find_library(MPI_LIBRARY
    NAMES mpich mpi_gnu mpi_nvidia mpi
    PATH_SUFFIXES lib64 lib
    PATHS ${_mpich_hints}
    ${_mpich_no_default}
)

find_package_handle_standard_args(MPICH
    REQUIRED_VARS
        MPI_LIBRARY
        MPI_INCLUDE_DIR
)

if (MPICH_FOUND)
    set(MPI_FOUND TRUE)
    mark_as_advanced(
        MPI_INCLUDE_DIR
        MPI_LIBRARY
    )

    if (NOT TARGET MPI::MPI)
        add_library(MPI::MPI INTERFACE IMPORTED GLOBAL)
        target_include_directories(MPI::MPI INTERFACE ${MPI_INCLUDE_DIR})
        target_link_libraries(MPI::MPI INTERFACE ${MPI_LIBRARY})
    endif()

    message(STATUS "Found MPICH (module): ${MPI_LIBRARY}")
else()
    message(FATAL_ERROR
        "MPICH not found.\n"
        "  Load the cray-mpich module (sets CRAY_MPICH_ROOTDIR), or\n"
        "  set MPI_ROOT to the installation prefix,\n"
        "  e.g. -DMPI_ROOT=/path/to/mpich"
    )
endif()
