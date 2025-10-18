
# this is a workaround for Perlmutter
include(FindPackageHandleStandardArgs)

set(hdf5_FOUND FALSE)

set(hdf5_SEARCH_PATHS 
    /usr
    /usr/local
    /opt
    /opt/local
    ${HDF5_ROOT}
    ${CMAKE_PREFIX_PATH}
   )

find_path(
    hdf5_INCLUDE_DIR
    NAMES 
        hdf5.h
    PATH_SUFFIXES 
        include
    PATHS 
        ${hdf5_SEARCH_PATHS}
)

find_library(
    hdf5_LIBRARY
    NAMES 
        hdf5
    PATH_SUFFIXES
        lib64 lib
    PATHS 
        ${hdf5_SEARCH_PATHS}
)

if (hdf5_INCLUDE_DIR AND hdf5_LIBRARY)
    set(hdf5_FOUND TRUE)
endif()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(hdf5
    REQUIRED_VARS 
    hdf5_LIBRARY 
    hdf5_INCLUDE_DIR
    hdf5_FOUND
) 

if(hdf5_FOUND AND NOT TARGET hdf5::hdf5)
    add_library(hdf5::hdf5 INTERFACE IMPORTED)
    set_property(TARGET hdf5::hdf5 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${hdf5_INCLUDE_DIR}")
    set_property(TARGET hdf5::hdf5 PROPERTY INTERFACE_LINK_LIBRARIES "${hdf5_LIBRARY}")
    mark_as_advanced(hdf5_LIBRARY)
    mark_as_advanced(hdf5_INCLUDE_DIR)
endif()
