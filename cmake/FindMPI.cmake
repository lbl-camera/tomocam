
# this is a workaround for Perlmutter
include(FindPackageHandleStandardArgs)

# enable C language
set(MPI_FOUND FALSE)
enable_language(C)

set(MPI_SEARCH_PATHS 
    /usr
    /usr/local
    /opt
    /opt/local
    ${MPI_ROOT}
    ${CMAKE_PREFIX_PATH}
   )

find_path(
    MPI_INCLUDE_DIR
    NAMES 
        mpi.h
    PATH_SUFFIXES 
        include
    PATHS 
        ${MPI_SEARCH_PATHS}
)

find_library(
    MPI_LIBRARY
    NAMES 
        libmpi
    PATH_SUFFIXES
        lib64 lib
    PATHS 
        ${MPI_SEARCH_PATHS}
)

if (MPI_INCLUDE_DIR AND MPI_LIBRARY)
    set(MPI_FOUND TRUE)
endif()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(MPI
    REQUIRED_VARS 
    MPI_LIBRARY 
    MPI_INCLUDE_DIR
    MPI_FOUND
) 

if(MPI_FOUND AND NOT TARGET MPI::MPI)
    add_library(MPI::MPI INTERFACE IMPORTED)
    set_property(TARGET MPI::MPI PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_INCLUDE_DIR}")
    set_property(TARGET MPI::MPI PROPERTY INTERFACE_LINK_LIBRARIES "${MPI_LIBRARY}")
    mark_as_advanced(MPI_LIBRARY)
    mark_as_advanced(MPI_INCLUDE_DIR)
endif()
