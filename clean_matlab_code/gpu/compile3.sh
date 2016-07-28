MW_NVCC_PATH=/usr/local/cuda/bin/nvcc
flags1='-Xcompiler -fPIC -I. -I/usr/local/MATLAB/R2015b/extern/include -I/usr/local/MATLAB/R2015b/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c'
flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2015b/bin/glnxa64/ -lmwgpu'
#flags1='-Xcompiler -fPIC -I/usr/local/MATLAB/R2015b/extern/include -I/usr/local/MATLAB/R2015b/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c'
#flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2015b/bin/glnxa64/ -lmwgpu'
/usr/local/cuda/bin/nvcc ${flags1} polarsample_transpose.cu
/usr/local/MATLAB/R2015b/bin/mex  -cxx polarsample_transpose.o  ${flags2}

