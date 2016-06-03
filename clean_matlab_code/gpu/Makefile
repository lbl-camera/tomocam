CC = nvcc
MATLAB := /usr/local/MATLAB/R2015b
MEX = $(MATLAB)/bin/mex

CFLAGS =-c -O2 -g -Xcompiler -fPIC -m64 -arch sm_35 -DNDEBUG
CPPFLAGS =-I$(MATLAB)/extern/include -I$(MATLAB)/toolbox/distcomp/gpu/extern/include
LDFLAGS =-L/usr/local/cuda/lib64 -L$(MATLAB)/bin/glnxa64
LIBS = -lcudart -lmwgpu


SOURCES=polarsample.cu polarbin1.cu polargrid_cub.cu 
#gcoo2csr.cu gcsr2coo.cu
#SOURCES = polargrid_cub.cu 
OBJECTS=$(SOURCES:.cu=.o)
MEXFILE=$(SOURCES:.cu=.mexa64)
all: $(OBJECTS) $(MEXFILE)


polarsample.mexa64: polarsample.o cuda_sample.o
	$(MEX) -cxx $(LDFLAGS) $(LIBS) $?

polarbin1.mexa64: polarbin1.o
	$(MEX) -cxx $(LDFLAGS) $(LIBS) $<
	
polargrid_cub.mexa64: polargrid_cub.o
	$(MEX) -cxx $(LDFLAGS) $(LIBS) $<

%.o: %.cu
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@


polargrid_cub.o: polargrid2.h

clean:
	rm -f $(OBJECTS) $(MEXFILE)
