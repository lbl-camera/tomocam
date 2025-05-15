from setuptools import Extension, setup
from setuptools_cpp import CMakeExtension, ExtensionBuilder

extensions = [ CMakeExtension(name="tomocam.cTomocam", sourcedir=".") ]

setup(
    ext_modules=extensions,
    cmdclass=dict(build_ext=ExtensionBuilder),
    zip_safe=False,
    )

