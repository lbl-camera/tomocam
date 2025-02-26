import os
import platform
import re
import subprocess
import sys
import sysconfig
from typing import Any, Dict

from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
from setuptools_cpp import CMakeExtension, ExtensionBuilder


def build(setup_kwargs: Dict[str, Any] = {}) -> None:
    srcdir = os.path.abspath(os.path.dirname(__file__))
    print("srcdir: ", srcdir)
    ext_modules = [CMakeExtension("tomocam.cTomocam", sourcedir=srcdir)]
    setup_kwargs.update({
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        })

