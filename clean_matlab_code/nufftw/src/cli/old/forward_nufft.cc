/* Michal Zarrouk, June 2013 */

#include "nufft.h"
//#include <cstring>

int main(int argc, char* argv[])
{
	// usage: forward_nufft trajfilename Ndim imagesize imagefilename kdatafilename Nthreads maxaliasingerror_power alpha resampling_method

	// get input data
	char* trajfilename = argv[1];
    
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
    
    char* imagefilename = argv[3+Ndim];
	
	char* kdatafilename = argv[4+Ndim];
	
	int Nthreads = atoi(argv[5+Ndim]);
	
	DATA_T maxaliasingerror = pow(10.0,atof(argv[6+Ndim]));
	
	DATA_T alpha = atof(argv[7+Ndim]);

	resampling_method_t resampling_method = get_resampling_method(argv[8+Ndim]);
	
	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	
	
	cerr << "**nuFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";

	// set number of threads for OpenMP	
	omp_set_num_threads(Nthreads);
	
	// load/create transform implementation data
	if (resampling_method == PRECOMPUTEDCONV)
		oski_Init();
#if DATA_TYPE == DOUBLE
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
#elif DATA_TYPE == FLOAT
	fftwf_init_threads();
	fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
	nuFFT_implementation_t nuFFT_imp(trajfilename, Ndim, imagesize, maxaliasingerror, alpha, resampling_method, fftw_flags, Nthreads);
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	read_data_file(imagefilename, cartesian_data); // possible: use Ndata=direct_nuDFT_imp.Ntotal
	noncartesian_data = (CPLX_T*) calloc (nuFFT_imp.Nsamples, sizeof(CPLX_T));

	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	nuFFT_imp.forward(cartesian_data, noncartesian_data);
	cerr << timestamp()-t1 << "s\n";
	
	
	// write transformed (target) data to file
	write_data_file(kdatafilename, noncartesian_data, nuFFT_imp.Nsamples);
	
	nuFFT_imp.Free();
	free(imagesize);
	free(noncartesian_data);
	free(cartesian_data);
	fftw_cleanup_threads();
	oski_Close();
	return 0;
}

