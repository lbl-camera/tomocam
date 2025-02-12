
include(FindPackageHandleStandardArgs)

set(nlohmann_json_SEARCH_PATHS
    ~/json
    ~/nlohmann_json
    /usr/local
    /opt/local
    /usr
    /opt
    ${CMAKE_PREFIX_PATH}
)

find_path(nlohmann_json_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS $ENV{nlohmann_json_DIR}
    PATH_SUFFIXES include
    PATHS ${nlohmann_json_SEARCH_PATHS}
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(nlohmann_json DEFAULT_MSG nlohmann_json_INCLUDE_DIR)

if(nlohmann_json_INCLUDE_DIR)
    mark_as_advanced(nlohmann_json_INCLUDE_DIR)
    mark_as_advanced(nlohmann_json_FOUND)
endif()

if (nlohmann_json_FOUND AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_property(TARGET nlohmann_json::nlohmann_json PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_INCLUDE_DIR}")
endif()
