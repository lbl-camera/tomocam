# -------------------------------------------------------------------------------
# Tomocam Copyright (c) 2018
#
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals from the
# U.S. Dept. of Energy). All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at
# IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.
#---------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)
project(tomocam CXX CUDA) 
set(tomocam_VERSION_MAJOR 2) 
set(tomocam_VERSION_MINOR 5)

#C++ 11 Stuff
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-DTOMOCAM_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")


option(ENABLE_TESTS "Enable tests" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(SRC 
    ${PROJECT_SOURCE_DIR}/src/backward.cpp 
    ${PROJECT_SOURCE_DIR}/src/forward.cpp
    ${PROJECT_SOURCE_DIR}/src/staging.cpp
    ${PROJECT_SOURCE_DIR}/src/iradon.cpp
    ${PROJECT_SOURCE_DIR}/src/radon.cpp
    ${PROJECT_SOURCE_DIR}/src/gradient.cpp
    ${PROJECT_SOURCE_DIR}/src/tvd_update.cpp
    ${PROJECT_SOURCE_DIR}/src/mbir.cpp
   )

set(CUSRC 
    ${PROJECT_SOURCE_DIR}/src/nufft_utils.cu
    ${PROJECT_SOURCE_DIR}/src/cuTVD.cu
    ${PROJECT_SOURCE_DIR}/src/fftshift.cu
    ${PROJECT_SOURCE_DIR}/src/rescale.cu
    ${PROJECT_SOURCE_DIR}/src/calc_error.cu
    ${PROJECT_SOURCE_DIR}/src/tv_hessian.cu
    ${PROJECT_SOURCE_DIR}/src/padding.cu
    ${PROJECT_SOURCE_DIR}/src/psf.cu
   )

include_directories(${PROJECT_SOURCE_DIR}/src)

# CUDA Stuff
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${CMAKE_CUDA_TOOKIT_LIB_DIRECTORIES})
set(CMAKE_CUDA_FLAGS "--default-stream per-thread ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -DTOMOCAM_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG}")

#python
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED) 
include_directories(${Python_INCLUDE_DIRS})

#pybind11
find_package(pybind11 REQUIRED) 
include_directories(${pybind11_INCLUDE_DIR})
option(USING_SETUP_PY "building through setup.py" OFF)
if (USING_SETUP_PY)
    add_definitions(-DUSE_PYBIND11_PRINT)
endif(USING_SETUP_PY)

# openmp
find_package(OpenMP) 
if (OpenMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif(OpenMP_FOUND)

find_package(cufinufft REQUIRED)
include_directories(${cufinufft_INCLUDE_DIR})

add_library(dummy STATIC ${CUSRC}) 
set_target_properties(dummy PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON 
                      CUDA_VISIBILITY_PRESET "hidden"
                     )
set_property(TARGET dummy PROPERTY CUDA_ARCHITECTURES 60)

add_library(cTomocam MODULE src/pyTomocam.cpp ${SRC})
set_target_properties(cTomocam PROPERTIES 
                      CXX_VISIBILITY_PRESET "hidden" 
                      PREFIX "${PYTHON_MODULE_PREFIX}" 
                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(cTomocam PRIVATE cufft)
target_link_libraries(cTomocam PRIVATE dummy)
target_link_libraries(cTomocam PRIVATE pybind11::module)
target_link_libraries(cTomocam PRIVATE ${Python_LIBRARIES})
target_link_libraries(cTomocam PUBLIC ${cufinufft_LIBRARY})
target_compile_features(cTomocam PRIVATE cxx_std_17)

install(TARGETS cTomocam 
        ARCHIVE DESTINATION tomocam
        LIBRARY DESTINATION tomocam)



if (${ENABLE_TESTS})
    add_library(tomocam STATIC ${SRC})
    target_link_libraries(tomocam PRIVATE dummy cufft)
    target_link_libraries(tomocam PRIVATE pybind11::module)
    target_link_libraries(tomocam PRIVATE ${Python_LIBRARIES})
    target_compile_features(tomocam PRIVATE cxx_std_17)
    add_subdirectory(test)
endif()
