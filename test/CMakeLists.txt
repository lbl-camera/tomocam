
include_directories(${CMAKE_SOURCE_DIR}/src)



find_package(hdf5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})

# JSON
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(padding padding.cpp)
target_link_libraries(padding PRIVATE tomocam)
target_link_libraries(padding PRIVATE ${HDF5_LIBRARY})
target_link_libraries(padding PRIVATE nlohmann_json::nlohmann_json)


add_executable(partition partition.cpp)
target_link_libraries(partition PRIVATE ${HDF5_LIBRARY})
target_link_libraries(partition PRIVATE ${MPI_C_LIBRARIES})

# test dropcol
add_executable(dropcol dropcol.cpp)
target_link_libraries(dropcol PRIVATE tomocam)
target_link_libraries(dropcol PRIVATE ${HDF5_LIBRARY})
target_link_libraries(dropcol PRIVATE nlohmann_json::nlohmann_json)


add_executable(read reader.cpp)
target_link_libraries(read PRIVATE ${HDF5_LIBRARY})
target_link_libraries(read PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(read PRIVATE nlohmann_json::nlohmann_json)

# backproj
add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomocam)
target_link_libraries(backproj PRIVATE ${HDF5_LIBRARY})
target_link_libraries(backproj PRIVATE nlohmann_json::nlohmann_json)

# gradient
add_executable(gradient gradient.cpp)
target_link_libraries(gradient PRIVATE tomocam)
target_link_libraries(gradient PRIVATE ${HDF5_LIBRARY})
target_link_libraries(gradient PRIVATE nlohmann_json::nlohmann_json)

add_executable(tv tvd.cpp)
target_link_libraries(tv PRIVATE tomocam)
target_link_libraries(tv PRIVATE ${HDF5_LIBRARY})

# test for the forward projection
add_executable(forward forward.cpp)
target_link_libraries(forward PRIVATE tomocam)
target_link_libraries(forward PRIVATE ${HDF5_LIBRARY})


# test for Toeplitz matrix
add_executable(toeplitz toeplitz.cpp)
target_link_libraries(toeplitz PRIVATE tomocam)
target_link_libraries(toeplitz PRIVATE ${HDF5_LIBRARY})

# test for error calculation
add_executable(error error.cpp)
target_link_libraries(error PRIVATE tomocam)
target_link_libraries(error PRIVATE ${HDF5_LIBRARY})

# MBIR
add_executable(mbir mbir.cpp)
target_link_libraries(mbir tomocam)
target_link_libraries(mbir ${HDF5_LIBRARY})
target_link_libraries(mbir nlohmann_json::nlohmann_json)
target_link_libraries(mbir ${MPI_C_LIBRARIES})
