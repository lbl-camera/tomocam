FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set noninteractive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    curl \
    wget \
    build-essential \
    clang \
    libomp-dev \
    ninja-build \
    libopenblas-dev \
    gfortran \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libfftw3-dev \
    pybind11-dev \
    libopenmpi-dev \
    libtbb-dev \
    python3-numpy \
    python3-skbuild \
    python3-pyfftw \
    python3-pywt \
    python3-scipy \
    python3-six \
    python3-numexpr \
    python3-skimage \
    python3-skimage-lib \
    python3-tifffile \
    python3-h5py \
    python3-importlib-metadata \
    python3-opencv \
    python3-pandas \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Install newer CMake version
RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh && \
    sh cmake.sh --prefix=/usr/local --skip-license && \
    rm cmake.sh

# Set clang as the default C/C++ compiler using update-alternatives
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

# Set Python 3.11 as the default python3 and install/upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3.11 -m pip install --upgrade pip

# clone tomopy and install python part of the code
RUN git clone https://github.com/tomopy/tomopy.git && \
    cd tomopy && python3 -m pip install . --no-deps

# build the c part and instal
RUN cd tomopy && \
    cmake -S . -B build  -GNinja  \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DTOMOPY_USE_MKL=OFF && \
    cmake --build build  &&  cmake --install build 
RUN rm -rf tomopy

# Install finufft
RUN git clone https://github.com/flatironinstitute/finufft.git && \
    cd finufft && \
    cmake -S . -B build -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr/local  \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS:BOOL=OFF \
        -DBUILD_TESTING:BOOL=OFF \
        -DBUILD_EXAMPLES:BOOL=OFF \
        -DFINUFFT_USE_CUDA:BOOL=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DFINUFFT_STATIC_LINKING:BOOL=OFF && \
    cmake --build build && cmake --install build
RUN rm -rf finufft

# Install tomocam from GitHub (perlmutter branch)
RUN git clone -b perlmutter https://github.com/lbl-camera/tomocam.git && \
    cd tomocam && python3 -m pip install . && cd .. && rm -rf tomocam


#WORKDIR /app



# COPY ./operators/832/recon/run.py ./run.py
# COPY ./operators/832/recon/processor.py ./processor.py

# Set the entrypoint to use Python 3.11 explicitly
# ENTRYPOINT ["python3.11", "run.py"]
