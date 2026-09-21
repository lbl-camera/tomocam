#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NERSC_MODS=NERSC_MODULES.sh

if [ -f $SCRIPT_DIR/$NERSC_MODS ]; then
    . $SCRIPT_DIR/$NERSC_MODS
else
    echo "Environment file not found"
fi
CC=`which gcc`
CXX=`which g++`
cmake -S . -B build -G Ninja  \
    -DCMAKE_BUILD_TYPE=Release  \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_CUDA_HOST_COMPILER=$CXX \
    -DENABLE_TESTS:BOOL=ON \
    -Dfinufft_DIR=$HOME/finufft \
    -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"

cmake --build build --target all -j 16
