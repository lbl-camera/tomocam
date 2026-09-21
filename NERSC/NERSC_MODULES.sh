#! /bin/bash
function module_load() {
    local mod="$1"

    # if module is alread loaded return
    if module is-loaded "$mod" 2> /dev/null; then
        return 0
    fi
    module load "$mod"
}

module_load cudatoolkit
module_load PrgEnv-gnu
module_load cpe-cuda
module_load cmake/3.30.2
module_load cray-hdf5
module_load cray-fftw

export CC=`which gcc`
export CXX=`which g++`

