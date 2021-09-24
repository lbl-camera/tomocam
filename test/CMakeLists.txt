
include_directories(${CMAKE_SOURCE_DIR}/src)



find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})


add_executable(tvd test_tvd.cpp)
target_link_libraries(tvd tomocam)
add_test(test_tvd tvd)

add_executable(read test_reader.cpp)
target_link_libraries(read ${HDF5_LIBRARIES})
add_test(test_reader read)

add_executable(mbir test_mbir.cpp)
target_link_libraries(mbir tomocam)
target_link_libraries(mbir ${HDF5_LIBRARIES})
add_test(test_mbir mbir)

add_executable(backward test_backward.cpp)
target_link_libraries(backward tomocam ${HDF5_LIBRARIES})
add_test(test_backward backward)

add_executable(forward test_forward.cpp)
target_link_libraries(forward tomocam)
add_test(test_forward forward)

add_executable(padding test_padding.cpp)
target_link_libraries(padding tomocam)
add_test(test_padding padding)
