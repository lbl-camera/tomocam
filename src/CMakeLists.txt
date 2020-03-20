
cmake_minimum_required(VERSION 3.8)
project(tomocam CXX CUDA) 
set(tomocam_VERSION_MAJOR 2) 
set(tomocam_VERSION_MINOR 0)

#C++ 11 Stuff
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-DTOMOCAM_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

set(SRC 
    ${PROJECT_SOURCE_DIR}/backward.cpp 
    ${PROJECT_SOURCE_DIR}/forward.cpp
    ${PROJECT_SOURCE_DIR}/staging.cpp
    ${PROJECT_SOURCE_DIR}/iradon.cpp
    ${PROJECT_SOURCE_DIR}/radon.cpp
    ${PROJECT_SOURCE_DIR}/gradient.cpp
    ${PROJECT_SOURCE_DIR}/tvd_update.cpp
   )

set(CUSRC 
    ${PROJECT_SOURCE_DIR}/cuPolarsampleTranspose.cu 
    ${PROJECT_SOURCE_DIR}/cuPolarsample.cu 
    ${PROJECT_SOURCE_DIR}/cuTVD.cu
    ${PROJECT_SOURCE_DIR}/fftshift.cu
    ${PROJECT_SOURCE_DIR}/deapod.cu
    ${PROJECT_SOURCE_DIR}/rescale.cu
    ${PROJECT_SOURCE_DIR}/calc_error.cu
   )

include_directories(${PROJECT_SOURCE_DIR}/src)

find_package(CUDA REQUIRED) 
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${CMAKE_CUDA_TOOKIT_LIB_DIRECTORIES})
set(LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
set(CMAKE_CUDA_FLAGS_DEBUG "-G -DTOMOCAM_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG}")

#python d
find_package(PythonLibs REQUIRED) 
include_directories(${PYTHON_INCLUDE_DIRS})

#pybind11
find_package(pybind11 REQUIRED) 
list(APPEND LIBS pybind11::module)

find_package(OpenMP) 
if (OpenMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif(OpenMP_FOUND)

add_library(dummy STATIC ${CUSRC}) 
set_target_properties(dummy PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON 
                      CUDA_VISIBILITY_PRESET "hidden"
                     )

#
#add_library(cTomocam MODULE ${SRC})
#set_target_properties(cTomocam PROPERTIES 
#                      CXX_VISIBILITY_PRESET "hidden" 
#                      PREFIX "${PYTHON_MODULE_PREFIX}" 
#                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
#target_link_libraries(cTomocam PRIVATE ${LIBS})
#target_link_libraries(cTomocam PRIVATE dummy)


# for testing
add_library(tomocam ${SRC})
set_target_properties(tomocam PROPERTIES CXX_VISIBILITY_PRESET "hidden")
target_link_libraries(tomocam PRIVATE ${LIBS})
target_link_libraries(tomocam PRIVATE dummy)

add_executable(gridrec "test_gridrec.cpp")
target_link_libraries(gridrec tomocam)

add_executable(mbir "test_mbir.cpp")
target_link_libraries(mbir tomocam)



#install(TARGETS cTomocam 
#        ARCHIVE DESTINATION tomocam
#        LIBRARY DESTINATION tomocam)
