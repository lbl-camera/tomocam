
include_directories(${CMAKE_SOURCE_DIR}/src)
# JSON
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(padding padding.cpp)
target_link_libraries(padding PRIVATE tomocam)
target_link_libraries(padding PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(padding PRIVATE nlohmann_json::nlohmann_json)


add_executable(partition partition.cpp)
target_link_libraries(partition PRIVATE tomocam)
target_link_libraries(partition PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(partition PRIVATE ${MPI_C_LIBRARIES})

# test dropcol
add_executable(dropcol dropcol.cpp)
target_link_libraries(dropcol PRIVATE tomocam)
target_link_libraries(dropcol PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(dropcol PRIVATE nlohmann_json::nlohmann_json)

find_package(nlohmann_json REQUIRED)

add_executable(read reader.cpp)
target_link_libraries(read PRIVATE tomocam)
target_link_libraries(read PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(read PRIVATE nlohmann_json::nlohmann_json)

# backproj
add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomocam)
target_link_libraries(backproj PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backproj PRIVATE nlohmann_json::nlohmann_json)

# gradient
add_executable(gradient gradient.cpp)
target_link_libraries(gradient PRIVATE tomocam)
target_link_libraries(gradient PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(gradient PRIVATE nlohmann_json::nlohmann_json)

add_executable(tv tvd.cpp)
target_link_libraries(tv PRIVATE tomocam)

# test for the forward projection
add_executable(forward forward.cpp)
target_link_libraries(forward PRIVATE tomocam)
target_link_libraries(forward PRIVATE ${HDF5_LIBRARIES})


# test for Toeplitz matrix
add_executable(toeplitz toeplitz.cpp)
target_link_libraries(toeplitz PRIVATE tomocam)
target_link_libraries(toeplitz PRIVATE ${HDF5_LIBRARIES})

# test for error calculation
add_executable(error error.cpp)
target_link_libraries(error PRIVATE tomocam)
target_link_libraries(error PRIVATE ${HDF5_LIBRARIES})

# MBIR
add_executable(mbir mbir.cpp)
target_link_libraries(mbir tomocam)
target_link_libraries(mbir ${HDF5_LIBRARIES})
target_link_libraries(mbir nlohmann_json::nlohmann_json)
target_link_libraries(mbir ${MPI_C_LIBRARIES})

# MBIR2
add_executable(mbir2 mbir2.cpp)
target_link_libraries(mbir2 tomocam)
target_link_libraries(mbir2 ${HDF5_LIBRARIES})
target_link_libraries(mbir2 nlohmann_json::nlohmann_json)
target_link_libraries(mbir2 ${MPI_C_LIBRARIES})
