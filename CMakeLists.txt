
cmake_minimum_required(VERSION 3.8)
project(tomocam CXX CUDA) 
set(tomocam_VERSION_MAJOR 2) 
set(tomocam_VERSION_MINOR 0)

#C++ 11 Stuff
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-DTOMOCAM_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

set(SRC 
    ${PROJECT_SOURCE_DIR}/src/norm2.cpp
    ${PROJECT_SOURCE_DIR}/src/caxpy.cpp
    ${PROJECT_SOURCE_DIR}/src/backward.cpp 
    ${PROJECT_SOURCE_DIR}/src/forward.cpp
    ${PROJECT_SOURCE_DIR}/src/staging.cpp
    ${PROJECT_SOURCE_DIR}/src/iradon.cpp
    ${PROJECT_SOURCE_DIR}/src/radon.cpp
    ${PROJECT_SOURCE_DIR}/src/gradient.cpp
    ${PROJECT_SOURCE_DIR}/src/tvd_update.cpp
   )

set(CUSRC 
    ${PROJECT_SOURCE_DIR}/src/cuPolarsampleTranspose.cu 
    ${PROJECT_SOURCE_DIR}/src/cuPolarsample.cu 
    ${PROJECT_SOURCE_DIR}/src/cuTVD.cu
    ${PROJECT_SOURCE_DIR}/src/fftshift.cu
    ${PROJECT_SOURCE_DIR}/src/deapod.cu
    ${PROJECT_SOURCE_DIR}/src/rescale.cu
    ${PROJECT_SOURCE_DIR}/src/calc_error.cu
   )

include_directories(${PROJECT_SOURCE_DIR}/src)

# CUDA Stuff
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${CMAKE_CUDA_TOOKIT_LIB_DIRECTORIES})
set(CMAKE_CUDA_FLAGS_DEBUG "-G -DTOMOCAM_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG}")

#python d
find_package(PythonLibs REQUIRED) 
include_directories(${PYTHON_INCLUDE_DIRS})

#pybind11
find_package(pybind11 REQUIRED) 
include_directories(${pybind11_INCLUDE_DIR})

find_package(OpenMP) 
if (OpenMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif(OpenMP_FOUND)

add_library(dummy STATIC ${CUSRC}) 
set_target_properties(dummy PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON 
                      CUDA_VISIBILITY_PRESET "hidden"
                     )


add_library(cTomocam MODULE src/pyTomocam.cpp ${SRC})
set_target_properties(cTomocam PROPERTIES 
                      CXX_VISIBILITY_PRESET "hidden" 
                      PREFIX "${PYTHON_MODULE_PREFIX}" 
                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(cTomocam PRIVATE cublas cufft)
target_link_libraries(cTomocam PRIVATE dummy)
target_link_libraries(cTomocam PRIVATE pybind11::module)

install(TARGETS cTomocam 
        ARCHIVE DESTINATION tomocam
        LIBRARY DESTINATION tomocam)
