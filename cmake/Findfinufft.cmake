include(FindPackageHandleStandardArgs)

set(finufft_SEARCH_PATHS
    ~/finufft
    /usr/local
    /opt/local
    /usr
    /opt
    ${finufft_PATH}
)

find_path(finufft_INCLUDE_DIR
    NAMES cufinufft.h 
    HINTS $ENV{finufft_DIR}
    PATH_SUFFIXES include
    PATHS ${finufft_SEARCH_PATHS}
)

find_library(finufft_LIBRARY
    NAMES cufinufft 
    HINTS $ENV{finufft_DIR}
    PATH_SUFFIXES lib64 lib
    PATHS ${finufft_SEARCH_PATHS}
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(finufft 
    REQUIRED_VARS 
    finufft_LIBRARY 
    finufft_INCLUDE_DIR
    HANDLE_COMPONENTS
) 

if(finufft_LIBRARY AND finufft_INCLUDE_DIR)
    mark_as_advanced(finufft_FOUND YES)
    mark_as_advanced(finufft_LIBRARY)
    mark_as_advanced(finufft_INCLUDE_DIR)
endif()

if (finufft_FOUND AND NOT TARGET finufft::finufft)
    add_library(finufft::finufft INTERFACE IMPORTED)
    set_property(TARGET finufft::finufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${finufft_INCLUDE_DIR}")
    set_property(TARGET finufft::finufft PROPERTY INTERFACE_LINK_LIBRARIES ${finufft_LIBRARY})
endif()
