from skbuild import setup as skbuild_setup

skbuild_setup(
    name="tomocam",
    ext_modules=[],
    packages=["tomocam"],
    include_package_data=True,
    cmake_args=["-DMULTI_PROC=ON", "-DENABLE_PYTHON=ON", "-DENABLE_TESTS=OFF"]
)
