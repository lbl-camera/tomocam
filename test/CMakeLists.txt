
include_directories(${CMAKE_SOURCE_DIR}/src)



find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})


add_executable(tvd test_tvd.cpp)
target_link_libraries(tvd tomocam)
add_test(test_tvd tvd)

add_executable(read test_reader.cpp)
target_link_libraries(read PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read PRIVATE ${Python_LIBRARIES})
add_test(test_reader read)

add_executable(mbir test_mbir.cpp)
target_link_libraries(mbir tomocam)
target_link_libraries(mbir ${HDF5_LIBRARIES})
target_link_libraries(mbir tomocam ${cufinufft_LIBRARY})
add_test(test_mbir mbir)

add_executable(backward test_backward.cpp)
target_link_libraries(backward tomocam ${HDF5_LIBRARIES})
target_link_libraries(backward tomocam ${cufinufft_LIBRARY})
add_test(test_backward backward)

add_executable(backproj test_backproj.cpp)
target_link_libraries(backproj tomocam)
target_link_libraries(backproj tomocam ${HDF5_LIBRARIES})
target_link_libraries(backproj tomocam ${cufinufft_LIBRARY})
add_test(test_backproj backproj)

add_executable(radon test_radon.cpp)
target_link_libraries(radon tomocam)
target_link_libraries(radon tomocam ${HDF5_LIBRARIES})
target_link_libraries(radon tomocam ${cufinufft_LIBRARY})
add_test(test_radon radon)
