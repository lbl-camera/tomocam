#! /bin/bash
## install nuFFTW and external library dependencies

## installation locations
nuFFTW_dir=$PWD
nuFFTW_bin_dir=$nuFFTW_dir/bin
nuFFTW_src_dir=$nuFFTW_dir/src
nuFFTW_include_dir=$nuFFTW_dir/include
nuFFTW_packages_dir=$nuFFTW_dir/packages
fftw_dir=$nuFFTW_packages_dir/fftw
fftw_build_double_dir=$fftw_dir/build-3.3.3-double
fftw_build_float_dir=$fftw_dir/build-3.3.3-float
oski_dir=$nuFFTW_packages_dir/oski
oski_build_dir=$oski_dir/build-1.0.1h
alldirs=(nuFFTW_dir nuFFTW_bin_dir nuFFTW_include_dir nuFFTW_packages_dir fftw_dir fftw_build_double_dir fftw_build_float_dir oski_dir oski_build_dir)

for DIR in ${alldirs[@]}
do
	eval "[ -d \$$DIR ] || mkdir \$$DIR"
done


## download external libraries
fftw_tarname=$fftw_dir/fftw-3.3.3.tar.gz
fftw_download_url='http://www.fftw.org/fftw-3.3.3.tar.gz'
if [ $(uname) == "Darwin" ]; then
	curl -o $fftw_tarname $fftw_download_url
elif [ $(uname) == "Linux" ]; then
	[ -e $fftw_tarname] || wget -O $fftw_tarname $fftw_download_url
fi
tar -xvzf $fftw_tarname -C $fftw_dir

oski_tarname=$oski_dir/oski-1.0.1h.tar.gz
oski_download_url='http://sourceforge.net/projects/oski/files/oski/1.0.1h/oski-1.0.1h.tar.gz'
if [ $(uname) == "Darwin" ]; then
	curl -o $oski_tarname -L $oski_download_url
else
	[ -e $oski_tarname] || wget -O $oski_tarname $oski_download_url
fi
tar -xvzf $oski_tarname -C $oski_dir


## install external libraries
if [ $(uname) == "Linux" ]; then
	fftwenableavx=--enable-avx
fi
# install fftw for double precision
cd $fftw_dir/fftw-3.3.3
./configure --prefix=$fftw_build_double_dir --enable-fma --enable-sse2 $fftwenableavx --enable-openmp --enable-threads CFLAGS="-Wall -O3 -g -fPIC"
fftw_build_double_logname=$fftw_build_double_dir/build-3.3.3-double.log
cp $fftw_dir/fftw-3.3.3/config.log $fftw_build_double_logname
make | tee -a $fftw_build_double_logname
make install | tee -a $fftw_build_double_logname

# install fftw for single precision
cd $fftw_dir/fftw-3.3.3
./configure --prefix=$fftw_build_float_dir  --enable-fma --enable-sse2 $fftwenableavx --enable-openmp --enable-threads --enable-float CFLAGS="-Wall -O3 -g -fPIC"
fftw_build_float_logname=$fftw_build_float_dir/build-3.3.3-float.log
cp $fftw_dir/fftw-3.3.3/config.log $fftw_build_float_logname
make | tee -a $fftw_build_float_logname
make install | tee -a $fftw_build_float_logname

# install oski
# If you don't have gfortran installed, oski will fail.
# This is because although oski does have a --disable-fortran option, for some reason it doesn't work.
# if you are on a mac, you can find gfortran and the necessary libraries to install it at http://cran.r-project.org/bin/macosx/tools/
# if you are on linux, you can get gfortran with "sudo apt-get install gfortran" (you will need administrator permissions).
if ! type -p gfortran > /dev/null ; then
	if [ $(uname) == "Darwin" ]; then
		curl -o ~/Downloads/gfortran-4.2.3.pkg "http://cran.r-project.org/bin/macosx/tools/gfortran-4.2.3.pkg"
		sudo installer -store -pkg ~/Downloads/gfortran-4.2.3.pkg -target /
	elif [ $(uname) == "Linux" ]; then
		sudo apt-get install gfortran
	fi
fi
# this will take a while.
cd $oski_dir/oski-1.0.1h
if [ $(uname) == "Darwin" ]; then
	oskiblasflags=(--with-blas="-framework vecLib" LDFLAGS="-DYA_BLAS -DYA_LAPACK -DYA_BLASMULT -lblas -llapack")
fi
./configure --prefix=$oski_build_dir --enable-int-single CFLAGS="-Wall -O3 -g" "${oskiblasflags[@]}"
oski_build_logname=$oski_build_dir/build-1.0.1h.log
cp $oski_dir/oski-1.0.1h/config.log $oski_build_logname
make | tee -a $oski_build_logname
make benchmarks | tee -a $oski_build_logname
make install | tee -a $oski_build_logname



# nuFFTW
# in order to compile the mex files, make sure the mex binary will be recoginzed:
# either set the path for the mex binary in the Makefile,
# or create a shortcut to it like this:
# sudo ln -s /usr/local/MATLAB/R2012a/bin/mex /usr/bin/mex

## write configuration file
nuFFTW_conf_filename=$nuFFTW_src_dir/nufftw.conf
if [ -f $nuFFTW_conf_filename ]; then
	rm $nuFFTW_conf_filename
fi
nufftwdirs=(nuFFTW_dir nuFFTW_bin_dir nuFFTW_include_dir fftw_build_double_dir fftw_build_float_dir oski_build_dir)
for DIR in ${nufftwdirs[@]}
do
	eval "echo ""$DIR = \$$DIR" >> "$nuFFTW_conf_filename"
done
cd $nuFFTW_src_dir
make

