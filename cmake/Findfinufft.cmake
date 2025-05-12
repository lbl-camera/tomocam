
# Find or Fetch the finufft library
#
# This module defines the following variables:
#   finufft_FOUND
#   finufft_INCLUDE_DIR
#   finufft_LIBRARIES
#
# This module defines the following targets:
#   finufft::finufft



include(FindPackageHandleStandardArgs)

# try to find the local finufft installation
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
        $ENV{finufft_DIR} 
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES 
        include
    PATHS 
        ${finufft_SEARCH_PATHS}
)

find_library(finufft_LIBRARIES
    NAMES 
        cufinufft 
    HINTS 
        $ENV{finufft_DIR} 
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES 
        lib64 
        lib
    PATHS 
        ${finufft_SEARCH_PATHS}
)

if (finufft_INCLUDE_DIR AND finufft_LIBRARIES)
    set(finufft_FOUND TRUE)


    FIND_PACKAGE_HANDLE_STANDARD_ARGS(finufft 
        REQUIRED_VARS 
            finufft_INCLUDE_DIR
            finufft_LIBRARIES
            finufft_FOUND
    )

    mark_as_advanced(
        finufft_INCLUDE_DIR
        finufft_LIBRARIES
    )

    # create imported target
    if (NOT TARGET finufft::finufft)
        add_library(finufft::finufft INTERFACE IMPORTED GLOBAL)
        target_include_directories(finufft::finufft INTERFACE ${finufft_INCLUDE_DIR})
        target_link_libraries(finufft::finufft INTERFACE ${finufft_LIBRARIES})
     endif()

     message(STATUS "Found finufft: ${finufft_LIBRARIES}")
else()
    message(FATAL_ERROR "local installation of finufft not found")
endif()

