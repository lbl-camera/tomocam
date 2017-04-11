
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

INCLUDE(FindCUDA)

ADD_DEFINITIONS(-DDEBUG)

LINK_DIRECTORIES(/usr/local/lib /usr/local/cuda/lib64)

CUDA_ADD_EXECUTABLE(polar polarsample.cc cuPolarsample.cu)
TARGET_LINK_LIBRARIES(polar -lafcuda -lcudart)

CUDA_ADD_EXECUTABLE(trans polarsample_transpose.cc cuPolarsampleTranspose.cu)
TARGET_LINK_LIBRARIES(trans -lafcuda -lcudart)

CUDA_ADD_EXECUTABLE(tvd test_tvd.cc cuTVD.cu)
TARGET_LINK_LIBRARIES(tvd -lafcuda -lcudart)

