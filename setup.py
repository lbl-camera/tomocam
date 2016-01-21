#! usr/bin/env ptyon

from __future__ import division, absolute_import, print_function
import os
import sys
import numpy.distutils.core

ext = Extension(name = 'gnufft/cGnufft',
                 sources = ['cuda/polarbin.cc', 'cuda/polargrid.cc' ],
                 extra_compile_args = [ cuda_inc ],
                 extra_link_args =  [ cuda_lib ]
                )

if __name__ == '__main__':
    setup(name = 'gnufft',
          description       = 'GPU Non-Uniform Faster Fourier Transform',
          version           = '1.0.0',
          author            = 'Dinesh Kumar',
          author_email      = 'dkumar@lbl.gov',
          packages          = [ 'gnufft' ],
          ext_modules       = [ext]
          )
# End of setup_example.py
