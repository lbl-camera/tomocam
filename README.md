# tomoCAM

Model-Based Iterative Reconstruction (MBIR) for synchrotron tomography data using Non-Uniform Fast Fourier Transforms (NUFFT) on GPUs.

![Container Build](https://github.com/lbl-camera/tomocam/actions/workflows/build-container.yml/badge.svg)

## Features

- GPU-accelerated reconstruction using CUDA and NUFFT
- Model-based iterative reconstruction with total variation regularization
- Multi-GPU support on single node
- Multi-node and multi-GPU support via MPI
- Python interface with C++/CUDA backend
- Docker container with complete environment

## Core Functions

- `tomocam.recon()` - MBIR reconstruction (multi-GPU on single node)
- `tomocam.recon_mpi()` - MBIR reconstruction (multi-GPU, multi-node via MPI)
- `tomocam.radon()` - Forward projection using NUFFT
- `tomocam.backproject()` - Back-projection using NUFFT

## Dependencies

- CUDA Toolkit
- CMake >= 3.20
- pybind11
- numpy
- [finufft](https://finufft.readthedocs.io/en/latest/install.html)

## Installation

### From Source

```bash
# Create virtual environment (recommended)
pip install virtualenv
virtualenv -p /usr/bin/python3 tomocam-venv
source tomocam-venv/bin/activate

# Install tomocam
git clone https://github.com/lbl-camera/tomocam.git
cd tomocam
pip install .
```

### Using Docker

The Docker image includes all dependencies and an MPI-enabled reconstruction pipeline:

```bash
# Pull the image
podman-hpc pull ghcr.io/lbl-camera/tomocam:latest

# Run reconstruction
srun -n 2 --mpi=pmi2 podman-hpc run --rm --openmpi-pmi2 --gpu \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  tomocam:perlmutter \
  --datadir /data/input \
  --filename data.h5 \
  --axis 512.5
```

The container expects HDF5 files in ALS 8.3.2 format and writes output as TIFF files.

### Building on Perlmutter (NERSC)

`setup.py` builds with `-DMULTI_PROC=ON`. On NERSC (`$NERSC_HOST` set) CMake then
requires the `cray-mpich` module and links against it, not against whatever `FindMPI`
finds. Some things to watch for:

**Modules.** Load `PrgEnv-gnu`, `cray-mpich`, `cudatoolkit`, `craype-accel-nvidia80`
and `cmake` before building (see `NERSC/NERSC_MODULES.sh`). If `CRAY_MPICH_ROOTDIR`
is not set, the configure step fails on purpose. Rebuild after changing any of these
modules.

**Don't install MPI from conda.** conda-forge's `mpi4py` pulls in `openmpi`, which
causes two problems:

- At build time, the env's `include/` has Open MPI's `mpi.h`. If that directory comes
  before cray-mpich's include directory, the sources compile against Open MPI and
  the link step fails with `undefined reference to 'ompi_mpi_comm_world'` (and
  `ompi_mpi_int`, `ompi_mpi_op_sum`, ...). `CMakeLists.txt` now puts cray-mpich's
  include directory first to guard against this.
- At run time, `mpi4py` loads Open MPI while `tomocam` loads Cray MPICH, and the two
  don't mix under `srun`.

Remove conda's MPI packages and build `mpi4py` against cray-mpich:

```bash
conda remove --force mpi4py openmpi mpi
MPICC="cc -shared" pip install --no-binary=mpi4py --no-cache-dir mpi4py
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # should report CRAY MPICH
```

Don't pass `--no-build-isolation` here: the build needs Cython, which pip installs in the
isolated build environment. A later `conda install`/`update` can bring `openmpi` back
if something depends on `mpi`; `conda install "mpich=*=external_*"` stops this, because
that package tells conda an MPI is already provided and installs no MPI files.

**Dependencies outside conda.** `finufft` (built with cuFINUFFT), `nlohmann_json` and
`pybind11` are found from `$HOME/finufft`, `$HOME/json` and `$HOME/pybind11` when
they aren't on `CMAKE_PREFIX_PATH`. HDF5 can resolve to the conda env's `libhdf5`
even when `cray-hdf5` is loaded. That works, but it means two HDF5 builds are in play.

**Stale builds.** scikit-build caches the CMake configuration in `_skbuild/`. Delete it
(`rm -rf _skbuild`) after changing modules or the conda env, otherwise `pip install .`
reuses the old include and library paths.

**Checking the result.**

```bash
ldd $(python -c "import tomocam, os; print(os.path.dirname(tomocam.__file__))")/cTomocam*.so | grep -i mpi
```

Every MPI library should be under `/opt/cray/pe/`, and nothing should be `libmpi.so.40`
(Open MPI).

**Citation**
------------
If you use this code in your research, please cite the following paper:

Kumar, D., Parkinson, D. Y. and Donatelli, J. J. (2024). *tomoCAM: fast model-based iterative reconstruction via GPU acceleration and non-uniform fast Fourier transforms*. **J. Synchrotron Rad**. 31, 85-94. https://doi.org/10.1107/S1600577523008962



**Copyright Notice**
---------------------

Tomocam Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

 

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.

 

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so. 

 

**License Agreement**
---------------------

Tomocam Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 

(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 

(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 

(3) Neither the name of the University of California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 

You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features, functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory, without imposing a separate written license agreement for such Enhancements, then you hereby grant the following license: a  non-exclusive, royalty-free perpetual license to install, use, modify, prepare derivative works, incorporate into other computer software, distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.
