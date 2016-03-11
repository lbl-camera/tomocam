MW_NVCC_PATH=/usr/local/cuda/bin/nvcc

flags1='-Xcompiler -fPIC -I/usr/local/MATLAB/R2015b/extern/include -I/usr/local/MATLAB/R2015b/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c'
flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/MATLAB/R2015b/bin/glnxa64/ -lmwgpu'

#flags2='-L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu'

/usr/local/cuda/bin/nvcc ${flags1} polargrid_cusp.cu
/usr/local/MATLAB/R2015a/bin/mex  -cxx polargrid_cusp.o   ${flags2}


#/usr/local/cuda/bin/nvcc ${flags1} polargrid_cusp.cu

#/usr/local/cuda/bin/nvcc ${flags1} polarsample.cu
#/usr/local/cuda/bin/nvcc ${flags1} cuda_sample.cu
#/usr/local/cuda/bin/nvcc ${flags1} polarbin1.cu
#/usr/local/cuda/bin/nvcc ${flags1} gspmv_csr.cu
#/usr/local/cuda/bin/nvcc ${flags1} gspmv_coo.cu
#/usr/local/cuda/bin/nvcc ${flags1} grow2ptr.cu
#/usr/local/cuda/bin/nvcc ${flags1} gptr2row.cu
#/usr/local/cuda/bin/nvcc ${flags1} gcoo2csr.cu
#/usr/local/cuda/bin/nvcc ${flags1} gcsr2coo.cu


#/usr/local/MATLAB/R2015b/bin/mex  -cxx polarsample.o  cuda_sample.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx polarbin1.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx gspmv_csr.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx gspmv_coo.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx grow2ptr.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx gptr2row.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx gcoo2csr.o ${flags2}
#/usr/local/MATLAB/R2015b/bin/mex -cxx gcsr2coo.o ${flags2}



#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG  -m64 -arch sm_35 -c polarsample.cu

#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c cuda_sample.cu

#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c gspmv_csr.cu
#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c gspmv_coo.cu
#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c gptr2row.cu
#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c grow2ptr.cu
#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c gcoo2csr.cu
#/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64 -arch sm_35 -c gcsr2coo.cu

#/usr/local/matlab/bin/mex -cxx polargrid_cusp.o  kernel.o -L/usr/local/MATLAB/R2014a/bin/glnxa64 -lcudart
#/usr/local/matlab/bin/mex -cxx polargrid_cusp.o  kernel.o -L/usr/local/cuda/lib64 -lcudart -lmx -lmex -lmat -lm
#/usr/local/matlab/bin/mex -cxx polargrid_cusp.o  kernel.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu



#/usr/local/matlab/bin/mex -cxx polarsample.o  cuda_sample.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx polarbin1.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_csr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gcsr2coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gcoo2csr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx grow2ptr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gptr2row.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/glnxa64/ -lmwgpu



#/usr/local/matlab/bin/mex -cxx polarsample.o  cuda_sample.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx polarbin1.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_csr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gcsr2coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gcoo2csr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gspmv_coo.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx grow2ptr.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu
#/usr/local/matlab/bin/mex -cxx gptr2row.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/matlab/bin/maci64/ -lmwgpu

