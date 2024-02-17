include(FindPackageHandleStandardArgs)


set(nlohmann_json_SEARCH_PATHS
    /usr/local
    /opt/local
    /usr
    /opt
    ~/json
)

find_path(nlohmann_json_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS $ENV{nlohmann_json_DIR}
    PATH_SUFFIXES include
    PATHS ${nlohmann_json_SEARCH_PATHS}
)


if(nlohmann_json_INCLUDE_DIR) 
    mark_as_advanced(nlohmann_json_FOUND YES)
endif()


find_package_handle_standard_args(nlohmann_json
    REQUIRED_VARS
    nlohmann_json_INCLUDE_DIR  
    HANDLE_COMPONENTS
)


if (nlohmann_json_FOUND AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    target_include_directories(nlohmann_json::nlohmann_json INTERFACE ${nlohmann_json_INCLUDE_DIR})
endif()
