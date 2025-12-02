
include_directories(${CMAKE_SOURCE_DIR}/src)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(padding padding.cpp)
target_link_libraries(padding PRIVATE tomocam)
target_link_libraries(padding PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(padding PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME padding_test COMMAND padding)


add_executable(partition partition.cpp)
target_link_libraries(partition PRIVATE tomocam)
target_link_libraries(partition PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(partition PRIVATE ${MPI_C_LIBRARIES})
add_test(NAME partition_test COMMAND partition)

add_executable(read reader.cpp)
target_link_libraries(read PRIVATE tomocam)
target_link_libraries(read PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(read PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME read_test COMMAND read)

# backproj
add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomocam)
target_link_libraries(backproj PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backproj PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME backproj_test COMMAND backproj)

# gradient
add_executable(gradient gradient.cpp)
target_link_libraries(gradient PRIVATE tomocam)
target_link_libraries(gradient PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(gradient PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME gradient_test COMMAND gradient)

add_executable(tv tvd.cpp)
target_link_libraries(tv PRIVATE tomocam)
add_test(NAME tv_test COMMAND tv)

# test for the forward projection
add_executable(forward forward.cpp)
target_link_libraries(forward PRIVATE tomocam)
target_link_libraries(forward PRIVATE ${HDF5_LIBRARIES})
add_test(NAME forward_test COMMAND forward)
# test for Toeplitz matrix
add_executable(toeplitz toeplitz.cpp)
target_link_libraries(toeplitz PRIVATE tomocam)
target_link_libraries(toeplitz PRIVATE ${HDF5_LIBRARIES})
add_test(NAME toeplitz_test COMMAND toeplitz)

# test for error calculation
add_executable(error error.cpp)
target_link_libraries(error PRIVATE tomocam)
target_link_libraries(error PRIVATE ${HDF5_LIBRARIES})
add_test(NAME error_test COMMAND error)


# test for the xerror calculation
add_executable(xerror xerror.cpp)
target_link_libraries(xerror PRIVATE tomocam)
add_test(NAME xerror_test COMMAND xerror)
