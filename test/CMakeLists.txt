
include_directories(${CMAKE_SOURCE_DIR}/src)



find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(nlohmann_json REQUIRED)

add_executable(tvd test_tvd.cpp)
target_link_libraries(tvd tomocam)
add_test(test_tvd tvd)

add_executable(read test_reader.cpp)
target_link_libraries(read PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read PRIVATE ${Python_LIBRARIES})
add_test(test_reader read)

add_executable(mbir test_mbir.cpp)
target_link_libraries(mbir PRIVATE tomocam)
target_link_libraries(mbir PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(mbir PRIVATE finufft::finufft)
target_link_libraries(mbir PRIVATE nlohmann_json::nlohmann_json)
add_test(test_mbir mbir)

add_executable(backward test_backward.cpp)
target_link_libraries(backward PRIVATE tomocam)
target_link_libraries(backward PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backward PRIVATE finufft::finufft)
add_test(test_backward backward)

add_executable(backproj test_backproj.cpp)
target_link_libraries(backproj PRIVATE tomocam)
target_link_libraries(backproj PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(backproj PRIVATE finufft::finufft)
add_test(test_backproj backproj)

add_executable(radon test_radon.cpp)
target_link_libraries(radon PRIVATE tomocam)
target_link_libraries(radon PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(radon PRIVATE finufft::finufft)
add_test(test_radon radon)
