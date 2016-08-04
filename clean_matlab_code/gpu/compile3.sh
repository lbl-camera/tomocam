MW_NVCC_PATH=/usr/local/cuda/bin/nvcc
flags1='-Xcompiler -fPIC -I. -I/usr/local/MATLAB/R2015a/extern/include -I/usr/local/MATLAB/R2015b/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c'
flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2015a/bin/glnxa64/ -lmwgpu'
#flags1='-Xcompiler -fPIC -I/usr/local/MATLAB/R2015a/extern/include -I/usr/local/MATLAB/R2015b/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c'
#flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2015b/bin/glnxa64/ -lmwgpu'


#/usr/local/cuda/bin/nvcc ${flags1} polarsample_transposev2.cu
#/usr/local/MATLAB/R2015a/bin/mex  -cxx polarsample_transposev2.o  ${flags2}

/usr/local/cuda/bin/nvcc ${flags1} polarsamplev2.cu
/usr/local/MATLAB/R2015a/bin/mex  -cxx polarsamplev2.o  ${flags2}

/usr/local/cuda/bin/nvcc ${flags1} kb.cu
/usr/local/MATLAB/R2015a/bin/mex  -cxx kb.o  ${flags2}
 
