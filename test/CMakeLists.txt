
include_directories(${CMAKE_SOURCE_DIR}/src)



find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# JSON
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(padding test_padding.cpp)
target_link_libraries(padding PRIVATE tomocam)
target_link_libraries(padding PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(padding PRIVATE nlohmann_json::nlohmann_json)
add_test(test_padding padding)


add_executable(partition test_partition.cpp)
target_link_libraries(partition PRIVATE ${HDF5_LIBRARIES})

# test dropcol
add_executable(dropcol test_dropcol.cpp)
target_link_libraries(dropcol PRIVATE tomocam)
target_link_libraries(dropcol PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(dropcol PRIVATE nlohmann_json::nlohmann_json)


add_executable(read test_reader.cpp)
target_link_libraries(read PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read PRIVATE ${Python_LIBRARIES})
target_link_libraries(read PRIVATE nlohmann_json::nlohmann_json)
add_test(test_reader read)

# backproj for device array
add_executable(backward test_backward.cpp)
target_link_libraries(backward PRIVATE tomocam)
target_link_libraries(backward PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backward PRIVATE nlohmann_json::nlohmann_json)
add_test(test_backward backward)


# backproj
add_executable(backproj test_backproj.cpp)
target_link_libraries(backproj PRIVATE tomocam)
target_link_libraries(backproj PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backproj PRIVATE nlohmann_json::nlohmann_json)
add_test(test_backproj backproj) 

# gradient
add_executable(gradient test_gradient.cpp)
target_link_libraries(gradient PRIVATE tomocam)
target_link_libraries(gradient PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(gradient PRIVATE nlohmann_json::nlohmann_json)

#[[
# total variation
add_executable(tv test_tvd.cpp)
target_link_libraries(tv PRIVATE tomocam)

add_executable(mbir test_mbir.cpp)
target_link_libraries(mbir tomocam)
target_link_libraries(mbir ${HDF5_LIBRARIES})
target_link_libraries(mbir ${finufft_LIBRARY})
add_test(test_mbir mbir)
]]

# test for the forward projection
add_executable(forward test_forward.cpp)
target_link_libraries(forward PRIVATE tomocam)
target_link_libraries(forward PRIVATE ${HDF5_LIBRARIES})


# test for Toeplitz matrix
add_executable(toeplitz test_toeplitz.cpp)
target_link_libraries(toeplitz PRIVATE tomocam)
target_link_libraries(toeplitz PRIVATE ${HDF5_LIBRARIES})

