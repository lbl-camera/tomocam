MW_NVCC_PATH=/usr/local/cuda/bin/nvcc

/usr/local/cuda-5.5/bin/nvcc -Xcompiler -fPIC -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -DNDEBUG -m64  -arch=sm_35 -rdc=true  -c  xgrid.cu
/usr/local/cuda-5.5/bin/nvcc --compiler-options -fPIC -arch=sm_35 -dlink -o xgrid_link.o xgrid.o  -lmwgpu -lcudadevrt -lcudart
/usr/local/matlab/bin/mex -cxx xgrid.o xgrid_link.o   -L/usr/local/matlab/bin/glnxa64/ -L/usr/local/cuda-5.5/lib64/ -lmwgpu -lcudadevrt -lcudart

