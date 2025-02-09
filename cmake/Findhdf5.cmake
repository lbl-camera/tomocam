
# this is a workaround for Perlmutter
include(FindPackageHandleStandardArgs)

set(hdf5_SEARCH_PATHS 
    ${HDF5_DIR}
    ${CMAKE_PREFIX_PATH}
   )

find_path(hdf5_INCLUDE_DIR
    NAMES hdf5.h
    HINTS $ENV{HDF5_DIR}
    PATH_SUFFIXES include
    PATHS ${hdf5_SEARCH_PATHS}
)

find_library(hdf5_LIBRARY
    NAMES hdf5
    HINTS $ENV{HDF5_DIR}
    PATH_SUFFIXES lib64 lib
    PATHS ${hdf5_SEARCH_PATHS}
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(hdf5
    REQUIRED_VARS 
    hdf5_LIBRARY 
    hdf5_INCLUDE_DIR
) 

if(hdf5_LIBRARY AND hdf5_INCLUDE_DIR)
    mark_as_advanced(hdf5_FOUND YES)
    mark_as_advanced(hdf5_LIBRARY)
    mark_as_advanced(hdf5_INCLUDE_DIR)
endif()
