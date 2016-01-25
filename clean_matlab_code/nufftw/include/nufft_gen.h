/* Michal Zarrouk, June 2013 */

/* This file includes general definitions for the nufft toolbox. */

#include <cstdlib>
#include <iostream>

using namespace std;

#include <omp.h>

/* type-dependent issues */
#define FLOAT 1
#define DOUBLE 2

// trajectory type
#if TRAJ_TYPE == DOUBLE
	#define TRAJ_T double
#elif TRAJ_TYPE == FLOAT
	#define TRAJ_T float
#endif

// data type
#if DATA_TYPE == DOUBLE
	#define DATA_T double
	#define INIT_FFTW \
		fftw_init_threads(); \
		fftw_plan_with_nthreads(omp_get_max_threads());
	#define CLEAN_FFTW \
		fftw_cleanup_threads();
	#define MEXDATA_CLASS mxDOUBLE_CLASS
#elif DATA_TYPE == FLOAT
	#define DATA_T float
	#define INIT_FFTW \
		fftwf_init_threads(); \
		fftwf_plan_with_nthreads(omp_get_max_threads());
	#define CLEAN_FFTW \
		fftwf_cleanup_threads();
	#define MEXDATA_CLASS mxSINGLE_CLASS
#endif
/* I know what you're thinking, that I should have used a macro for the FFTW type/name-mangling, something like

#define CPLX_T FFTW(complex)
#if DATA_TYPE == DOUBLE
	#define FFTW(name) fftw_ ## name
#elif DATA_TYPE == FLOAT
	#define FFTW(name) fftwf_ ## name
#endif

but I hate name-mangling macros and I swore to use them as less as possible, because they make the code unreadable.
*/




// complex data type
typedef DATA_T CPLX_T[2];


#include <math.h>
#include <sys/time.h>

#include "nufft_iofun.h" // input/output/file read & write functions


/* timestamp - returns time in seconds, for timing calculations. */
double timestamp()
{
	struct timeval tv;
	gettimeofday (&tv, 0);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}


void nufft_print(bool val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(short val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(unsigned short val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(int val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(unsigned int val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(long val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(unsigned long val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(float val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(double val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(long double val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(void* val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}

void nuFFT_print(streambuf* sb)
{
#ifdef NUFFT_PRINT
	cerr << sb;	
#endif
}

void nuFFT_print(ostream& (*pf)(ostream&))
{
#ifdef NUFFT_PRINT
	cerr << pf;	
#endif
}

void nuFFT_print(ios& (*pf)(ios&))
{
#ifdef NUFFT_PRINT
	cerr << pf;	
#endif
}

void nuFFT_print(ios_base& (*pf)(ios_base&))
{
#ifdef NUFFT_PRINT
	cerr << pf;	
#endif
}

void nuFFT_print(const char* val)
{
#ifdef NUFFT_PRINT
	cerr << val;	
#endif
}


void nuFFT_print_time(double elapsed_time)
{
	nuFFT_print(elapsed_time);
	nuFFT_print("s\n");
}

