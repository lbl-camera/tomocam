from skbuild import setup as skbuild_setup

skbuild_setup(
    name="tomocam",
    ext_modules=[],
    packages=["tomocam"],
    include_package_data=True,
    cmake_args=[
        "-GNinja",
        "-DMULTI_PROC=OFF",
        "-DENABLE_PYTHON=ON",
        "-DENABLE_TESTS=OFF",
        "-DCMAKE_CXX_COMPILER=g++",
        "-DCMAKE_CUDA_HOST_COMPILER=g++"
        ]
)
