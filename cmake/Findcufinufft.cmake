
set(cufinufft_SEARCH_PATHS
    /usr/local
    /opt/local
    /usr
    /opt
    /global/cfs/cdirs/m4055/finufft
    ~/finufft
)

find_path(cufinufft_INCLUDE_DIR
    NAMES cufinufft.h
    HINTS $ENV{cufinufft_DIR}
    PATH_SUFFIXES include
    PATHS ${cufinufft_SEARCH_PATHS}
)

find_library(cufinufft_LIBRARY
    NAMES cufinufft
    HINTS $ENV{cufinufft_DIR}
    PATH_SUFFIXES lib64 lib
    PATHS ${cufinufft_SEARCH_PATHS}
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(cufinufft REQUIRED_VARS cufinufft_LIBRARY cufinufft_INCLUDE_DIR)
