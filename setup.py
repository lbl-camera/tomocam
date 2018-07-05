import  os
from os.path import join

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = join(home, 'bin', 'nvcc')
    elif os.path.isdir('/usr/local/cuda'):
        home = '/usr/local/cuda'
        nvcc = join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    # setup config
    if os.path.isdir(join(home, 'include')):
        include = join(home, 'include')
    else:
        include = None

    if os.path.isdir(join(home, 'lib64')):
        libdir = join(home, 'lib64')
    elif os.path.isdir(join(home, 'lib')):
        libdir = join(home, 'lib')
    else:
        libdir = None

    cudaconfig = {
                    'home':home, 'nvcc':nvcc,
                    'include': include,
                    'lib64': libdir
                 }

    return cudaconfig
CUDA = locate_cuda()

src = ['cuda/pyGnufft.cpp', 'cuda/af_api.cpp', 'cuda/polarsample.cpp', 'cuda/tvd_update.cpp',\
    'cuda/polarsample_transpose.cpp', 'cuda/cuPolarsample.cu', 'cuda/cuPolarsampleTranspose.cu',\
    'cuda/cuTVD.cu']

src.append('cuda/debug.cu')
inc = ['cuda/pyGnufft.h', 'cuda/af_api.h', 'cuda/polarsample.h' ]

ext = Extension('tomocam.gnufft',
                sources = src,
                depends = inc,
                library_dirs=[CUDA['lib64']],
                libraries=['afcuda', 'cudart'],
                language='c++',
                runtime_library_dirs=[CUDA['lib64']],
                extra_compile_args=[ '-g', '-O0', '-DDEBUG' ],
                include_dirs = [CUDA['include']]
                )

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            #postargs = extra_postargs['nvcc']
            postargs = ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '-shared']
            postargs += [ '-std=c++11' ]
            postargs += [ '-g', '-O0', '-DDEBUG' ]
        else:
            postargs = []

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(name='tomocam',
      # random metadata. there's more you can supploy
      author='LBL Camera',
      version='1.0.1',
      ext_modules = [ext],
      packages = [ 'tomocam' ],
      cmdclass={'build_ext': custom_build_ext}
      )
