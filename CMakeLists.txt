
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

cmake_minimum_required(VERSION 3.20)

# Default to clang++ for C++ and CUDA if not already set
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER nvcc)
endif()

project(tomocam LANGUAGES CXX CUDA)
set(tomocam_VERSION_MAJOR 3) 
set(tomocam_VERSION_MINOR 0)

#C++ Stuff
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CUDA_STANDARD 20)

# debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

#  local CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Source files
set(SRC 
    # Core algorithms
    ${PROJECT_SOURCE_DIR}/src/core/backward.cpp 
    ${PROJECT_SOURCE_DIR}/src/core/iradon.cpp
    ${PROJECT_SOURCE_DIR}/src/core/forward.cpp
    ${PROJECT_SOURCE_DIR}/src/core/radon.cpp
    ${PROJECT_SOURCE_DIR}/src/core/gradient.cpp
    ${PROJECT_SOURCE_DIR}/src/core/gradient2.cpp
    ${PROJECT_SOURCE_DIR}/src/core/error.cpp
    ${PROJECT_SOURCE_DIR}/src/core/error2.cpp
    ${PROJECT_SOURCE_DIR}/src/core/xerror.cpp
    
    # Reconstruction algorithms
    ${PROJECT_SOURCE_DIR}/src/reconstruction/mbir.cpp
    ${PROJECT_SOURCE_DIR}/src/reconstruction/mbir2.cpp
    ${PROJECT_SOURCE_DIR}/src/reconstruction/tv_update.cpp
    
    # Preprocessing
    ${PROJECT_SOURCE_DIR}/src/preprocessing/postproc.cpp
    ${PROJECT_SOURCE_DIR}/src/preprocessing/preproc.cpp
    ${PROJECT_SOURCE_DIR}/src/preprocessing/padding.cpp
    
    
    # GPU kernels
    ${PROJECT_SOURCE_DIR}/src/gpu/cuTV2.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/gpu_ops.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/padding.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/fftshift.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/nufft_utils.cu
    ${PROJECT_SOURCE_DIR}/src/gpu/ramlak.cu
   )
include_directories(${PROJECT_SOURCE_DIR}/src)


# set flag for gprof
option(ENABLE_GPROF "Enable gprof" OFF)
mark_as_advanced(ENABLE_GPROF)
if (ENABLE_GPROF)
    message(STATUS "gprof is enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(ENABLE_GPROF)

# create a dependency interface shortcut
add_library(tomocam::deps INTERFACE IMPORTED)
target_link_libraries(tomocam::deps INTERFACE stdc++exp)

# CUDA Stuff
find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tomocam::deps INTERFACE CUDA::cudart)
target_link_libraries(tomocam::deps INTERFACE CUDA::cufft)
target_link_libraries(tomocam::deps INTERFACE CUDA::cuda_driver)
target_compile_definitions(tomocam::deps INTERFACE CUDA_API_PER_THREAD_DEFAULT_STREAM)

# openmp
find_package(OpenMP) 
if (OpenMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif(OpenMP_FOUND)

# Set option to use MPI
option(MULTI_PROC "Enable MPI" OFF)
message("-- MPI is ${MULTI_PROC}")
if (NOT MULTI_PROC)
    message("-- To enable MPI, run cmake with -DMULTI_PROC:BOOL=ON")
endif()

# MPI
if (MULTI_PROC)
    enable_language(C)
    find_package(MPI COMPONENTS C)
    if (MPI_FOUND)
        target_include_directories(tomocam::deps INTERFACE ${MPI_C_INCLUDE_PATH})
        target_compile_definitions(tomocam::deps INTERFACE MULTIPROC)
        target_link_libraries(tomocam::deps INTERFACE ${MPI_C_LIBRARIES})
    else()
        message(FATAL_ERROR "MPI not found")
    endif(MPI_FOUND)
endif(MULTI_PROC)


# finufft
find_package(finufft REQUIRED)
target_link_libraries(tomocam::deps INTERFACE finufft::finufft)

# TRY the official package
find_package(HDF5 REQUIRED COMPONENTS C)
target_link_libraries(tomocam::deps INTERFACE hdf5::hdf5)


find_package(TBB REQUIRED)
target_link_libraries(tomocam::deps INTERFACE TBB::tbb)

# build ojbect library
add_library(tomocam_objs OBJECT ${SRC})
set_target_properties(tomocam_objs PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON)
target_compile_features(tomocam_objs PUBLIC cxx_std_20)
target_link_libraries(tomocam_objs PUBLIC tomocam::deps)

# build shared library
add_library(tomocam SHARED $<TARGET_OBJECTS:tomocam_objs>)
set_target_properties(tomocam PROPERTIES 
                      VERSION ${tomocam_VERSION_MAJOR}.${tomocam_VERSION_MINOR} 
                      SOVERSION ${tomocam_VERSION_MAJOR})
target_link_libraries(tomocam PUBLIC tomocam::deps)
target_compile_features(tomocam PUBLIC cxx_std_20)

option(ENABLE_PYTHON "Enable python module" OFF)
if (ENABLE_PYTHON)
    message("-- Enabling python module ...")

    # find python
    set(Python_FIND_VIRTUALENV FIRST)
    set(PYTHON_EXECUTABLE python)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python_INCLUDE_DIRS})

    # find pybind11
    find_package(pybind11 REQUIRED)
    add_compile_definitions(USING_PYTHON)


    # build python module
    add_library(cTomocam MODULE src/bindings/pyTomocam.cpp $<TARGET_OBJECTS:tomocam_objs>)
    set_target_properties(cTomocam PROPERTIES 
                          CXX_VISIBILITY_PRESET "hidden" 
                          PREFIX "${PYTHON_MODULE_PREFIX}" 
                          SUFFIX "${PYTHON_MODULE_EXTENSION}")
    target_link_libraries(cTomocam PUBLIC tomocam::deps)
    target_compile_features(cTomocam PUBLIC cxx_std_20)
    install(TARGETS cTomocam DESTINATION tomocam)

else()
    message("-- Python module is disabled")
    message("-- To enable python module, run cmake with -DENABLE_PYTHON:BOOL=ON")
endif()


option(ENABLE_TESTS "Enable tests" OFF)
if (${ENABLE_TESTS})
    message("-- Enabling tests ...")
    enable_testing()
    add_subdirectory(test)
else()
    message("-- Tests are disabled")
    message("-- To enable tests, run cmake with -DENABLE_TESTS:BOOL=ON")
endif()

# Build recon executable
add_executable(recon ${PROJECT_SOURCE_DIR}/src/recon.cpp)
target_link_libraries(recon PUBLIC tomocam PRIVATE tomocam::deps)
target_compile_features(recon PUBLIC cxx_std_20)
