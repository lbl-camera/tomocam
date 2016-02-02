debug ?= 1

ifeq ($(nufft_type), DOUBLE)
	nufft_suffix = double
else ifeq ($(nufft_type), FLOAT)
	nufft_suffix = float
endif

include nufftw.conf

## compilers
cxx = g++
mex = mex


## pre-processor flags
# note that TRAJ_T and DATA_T don't have to be the same
includes = -I$(nuFFTW_include_dir)
defines = -DTRAJ_TYPE=$(nufft_type) -DDATA_TYPE=$(nufft_type) -DNUFFT_PRINT
cppflags += $(includes) $(defines)


## compiler flags
cxxflags = -fPIC -fopenmp
ifeq ($(debug),1)
	cxxflags += -g
endif


## linker flags
ldflags = -lm

## fftw
ifeq ($(nufft_type), DOUBLE)
	fftw_build_dir = $(fftw_build_double_dir)
else ifeq ($(nufft_type), FLOAT)
	fftw_build_dir = $(fftw_build_float_dir)
endif
fftwflags = -DMEASURE_TIME -I$(fftw_build_dir)/include -L$(fftw_build_dir)/lib -Wl,-rpath,$(fftw_build_dir)/lib
# the -Wl,-rpath flags are for runtime dynamic linking to the external libraries specified above
# and not the ones installed on mikgiant/usr/local, which would be used by default because they
# are first on it's LD_LIBRARY_PATH. I like using -rpath more than temporarily changing LD_LIBRARY_PATH.
ifeq ($(nufft_type), DOUBLE)
	fftwflags += -lfftw3 -lfftw3_omp
else ifeq ($(nufft_type), FLOAT)
	fftwflags += -lfftw3f -lfftw3f_omp
endif

## oski
oskiflags = -I$(oski_build_dir)/include -L$(oski_build_dir)/lib/oski -Wl,-rpath,$(oski_build_dir)/lib/oski `cat $(oski_build_dir)/lib/oski/site-modules-shared.txt`

## general nufft files
nufft_sources = tbessi.cpp
nuFFT_includes := nufft_gen.h nufft_iofun.h nufft_util.h nudft.h nufft.h nufft_gridding.h nufft_sparse.h nufftw.h
nuFFT_includes := $(addprefix $(nuFFTW_include_dir)/, $(nuFFT_includes))
nufft_dependencies = $(nuFFT_includes) $(nuFFT_sources)


nufft_all: nufft_all_double nufft_all_float

nufft_all_double:
	nufft_type=DOUBLE \
	make command_line_all_nufft mex_nufft

nufft_all_float:
	nufft_type=FLOAT \
	make command_line_all_nufft mex_nufft

#clean_nuFFTW:
#	rm $(bin)/nufftw



## command line targets
command_line_nudft_targets = nudft
command_line_nudft: $(command_line_nudft_targets)
$(command_line_nudft_targets): % : nufft_interface/%.cc $(nufft_dependencies)
	$(cxx) $< -o $(nuFFTW_bin_dir)/$@_$(nufft_suffix) $(cppflags) $(cxxflags) $(ldflags)

command_line_nufft_targets = create_nufft_impfile nufft
command_line_nufft: $(command_line_nufft_targets)
$(command_line_nufft_targets): % : nufft_interface/%.cc $(nufft_dependencies)
	$(cxx) $< -o $(nuFFTW_bin_dir)/$@_$(nufft_suffix) $(cppflags) $(cxxflags) $(ldflags) $(fftwflags) $(oskiflags) $(nufft_sources)

command_line_nufftw_targets = nufftw
command_line_nufftw: $(command_line_nufftw_targets)
$(command_line_nufftw_targets): % : nufft_interface/%.cc $(nuFFTW_include_dir)/%.h $(nufft_dependencies)
	$(cxx) $< -o $(nuFFTW_bin_dir)/$@_$(nufft_suffix) $(cppflags) $(cxxflags) $(ldflags) $(fftwflags) $(oskiflags) $(nufft_sources)

command_line_all_nufft_targets = $(command_line_nudft_targets) $(command_line_nufft_targets) $(command_line_nufftw_targets)
command_line_all_nufft: $(command_line_all_nufft_targets)


unamestr = $(shell uname)
ifeq ($(unamestr),Linux)
	mexLDFLAGS = LDFLAGS="$(LDFLAGS) -pthread -shared -fopenmp -undefined"
else ifeq ($(unamestr),Darwin)
	mexLDFLAGS = LDFLAGS="$(LDFLAGS) -fopenmp -undefined dynamic_lookup -bundle"
endif


## mex targets
mex_nufft_targets = nufft_mex nufftw_mex
mex_nufft: $(mex_nufft_targets)
$(mex_nufft_targets): % : mex/%.cc $(nufft_dependencies)
	$(mex) -c $< -outdir $(nuFFTW_bin_dir) $(cppflags) -g $(ldflags) $(fftwflags) $(oskiflags) $(nufft_sources) CFLAGS="$(CFLAGS) -fopenmp" $(mexLDFLAGS)
	bash -c "mv $(nuFFTW_bin_dir)/$@.o $(nuFFTW_bin_dir)/$@_$(nufft_suffix).o"
	bash -c "mv $(nuFFTW_bin_dir)/tbessi.o $(nuFFTW_bin_dir)/tbessi_$(nufft_suffix).o"
	$(mex) $(nuFFTW_bin_dir)/$@_$(nufft_suffix).o $(nuFFTW_bin_dir)/tbessi_$(nufft_suffix).o -outdir $(nuFFTW_bin_dir) -output $@_$(nufft_suffix) $(cppflags) -g -cxx $(ldflags) $(fftwflags) $(oskiflags) CFLAGS="$(CFLAGS) -fopenmp" $(mexLDFLAGS)


## command line nfft targets
nfft_dir ?= $(top_installation_dir)/nfft
nfft_build_dir ?= $(nfft_dir)/build-3.2.2
nfftflags = -I$(nfft_build_dir)/include -L$(nfft_build_dir)/lib -Wl,-rpath,$(nfft_build_dir)/lib -lfftw3_threads -lnfft3 -Wno-deprecated
command_line_nfft_targets = NFFT NFFT_direct
command_line_nfft: $(command_line_nfft_targets)
$(command_line_nfft_targets): % : nufft_interface/%.cc $(nuFFTW_include_dir)/nufft_util.h $(nuFFTW_include_dir)/nufft_iofun.h
	$(cxx) $< -o $(nuFFTW_bin_dir)/$@_$(nufft_suffix) $(cppflags) $(cxxflags) $(ldflags) $(fftwflags) $(nfftflags) $(nufft_sources)

