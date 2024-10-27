
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

#C++ Stuff
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_FLAGS_DEBUG "-DTOMOCAM_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

# CUDA Stuff
find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_FLAGS "--default-stream per-thread ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -DTOMOCAM_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

#  local CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


# Source files
set(SRC 
    ${PROJECT_SOURCE_DIR}/src/backward.cpp 
    ${PROJECT_SOURCE_DIR}/src/iradon.cpp
    ${PROJECT_SOURCE_DIR}/src/forward.cpp
    ${PROJECT_SOURCE_DIR}/src/radon.cpp
    ${PROJECT_SOURCE_DIR}/src/error.cpp
    ${PROJECT_SOURCE_DIR}/src/error2.cpp
    ${PROJECT_SOURCE_DIR}/src/gradient.cpp
    ${PROJECT_SOURCE_DIR}/src/gradient2.cpp
    ${PROJECT_SOURCE_DIR}/src/gpu/cuTVD.cu
    ${PROJECT_SOURCE_DIR}/src/tvd_update.cpp
    ${PROJECT_SOURCE_DIR}/src/mbir.cpp
    ${PROJECT_SOURCE_DIR}/src/mbir2.cpp
    ${PROJECT_SOURCE_DIR}/src/postproc.cpp
    ${PROJECT_SOURCE_DIR}/src/preproc.cpp
    ${PROJECT_SOURCE_DIR}/src/gpu/tv_hessian.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/gpu_ops.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/padding.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/fftshift.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/nufft_utils.cu
   )
include_directories(${PROJECT_SOURCE_DIR}/src)

#python
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED) 
include_directories(${Python_INCLUDE_DIRS})

#pybind11
find_package(pybind11 REQUIRED) 
include_directories(${pybind11_INCLUDE_DIR})
option(USING_SETUP_PY "building through setup.py" OFF)
if (USING_SETUP_PY)
    add_compile_definitions(USING_SETUP_PY)
endif(USING_SETUP_PY)

# openmp
find_package(OpenMP) 
if (OpenMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif(OpenMP_FOUND)


# MPI
enable_language(C)
find_package(MPI COMPONENTS C)
if (MPI_FOUND)
    include_directories(${MPI_C_INCLUDE_DIRS})
    add_compile_definitions(MULTIPROC)
endif(MPI_FOUND)

# finufft
find_package(finufft REQUIRED)
include_directories(${finufft_INCLUDE_DIR})

# hdf5
# TRY the official package
find_package(HDF5)
if (NOT HDF5_FOUND)
    # use environment variable HDF5_DIR
    find_package(hdf5 REQUIRED)
    set(HDF5_INCLUDE_DIRS ${hdf5_INCLUDE_DIR})
    set(HDF5_LIBRARIES ${hdf5_LIBRARY})
endif()
include_directories(${HDF5_INCLUDE_DIRS})

# build shared library
add_library(tomocam SHARED ${SRC})
target_link_libraries(tomocam PUBLIC finufft::finufft)
target_link_libraries(tomocam PUBLIC CUDA::cufft)
target_link_libraries(tomocam PUBLIC CUDA::cudart)
target_link_libraries(tomocam PUBLIC CUDA::cuda_driver)
target_link_libraries(tomocam PUBLIC ${HDF5_LIBRARIES})
target_link_libraries(tomocam PUBLIC ${MPI_C_LIBRARIES})
target_compile_features(tomocam PUBLIC cxx_std_20)


# build python module
add_library(cTomocam MODULE src/pyTomocam.cpp)
set_target_properties(cTomocam PROPERTIES 
                      CXX_VISIBILITY_PRESET "hidden" 
                      PREFIX "${PYTHON_MODULE_PREFIX}" 
                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(cTomocam PUBLIC tomocam)
target_link_libraries(tomocam PUBLIC pybind11::module)
target_link_libraries(tomocam PUBLIC ${Python_LIBRARIES})
target_compile_features(cTomocam PUBLIC cxx_std_20)


option(ENABLE_TESTS "Enable tests" OFF)
if (${ENABLE_TESTS})
    message("-- Enabling tests ...")
    add_subdirectory(test)
else()
    message("-- Tests are disabled")
    message("-- To enable tests, run cmake with -DENABLE_TESTS:BOOL=ON")
endif()
