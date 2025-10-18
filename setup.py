from skbuild import setup as skbuild_setup
                          '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + output_dir,
                          '-DCMAKE_BUILD_TYPE=' + build_type,
                          '-DPYBIND11_PYTHON_VERSION=' + pyver,
                          '-DPYTHON_LIBRARY=' + pylib,
                          '-DPYTHON_INCLUDE_DIR=' + pyinc
                         ]
            cmake_args.extend([x for x in os.environ.get('CMAKE_COMMON_VARIABLES', '').split(' ') if x])

            env = os.environ.copy()
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['make', '-j'], cwd=self.build_temp, env=env)
            print()
        else:
            super().build_extension(ext)

skbuild_setup(
    name="tomocam",
    ext_modules=[],
    packages=["tomocam"],
    include_package_data=True
)
