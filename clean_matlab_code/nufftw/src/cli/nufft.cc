/* Michal Zarrouk, June 2013 */

#include "nufft.h"

nuFFT_implementation_t get_implementation(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	/* usage:
		nufft Nthreads direction imagefilename kdatafilename Ndim imagesize maxerr_power alpha resamplingmethod trajfilename
		nufft Nthreads direction imagefilename kdatafilename impfilename
	*/

	nuFFT_print("** nuFFT **\n");
	
	int Nthreads = atoi(argv[1]);
	omp_set_num_threads(Nthreads);
	
	nuDFT_direction_t nuFFT_direction = get_nuDFT_direction(argv[2]);
	
	char* imagefilename = argv[3];
	
	char* kdatafilename = argv[4];

	double t0 = timestamp();
	nuFFT_print("transform initialization and data loading: \n");

	nuFFT_implementation_t nuFFT_imp = get_implementation(argc, argv);
	
	nuFFT_print("\t");
	nuFFT_init(nuFFT_imp.resampling_method);
	
	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	nuFFT_print("\t");
	nuFFT_imp.tune(fftw_flags, Nthreads);
	
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	
	if (nuFFT_direction == ADJOINT) {
		CPLX_T* noncartesian_data_nodcf = read_data_file(kdatafilename); // possible: use Ndata=nuFFT_imp.Nsamples
		noncartesian_data = density_compensate(noncartesian_data_nodcf, nuFFT_imp.sqrtdcf, nuFFT_imp.Nsamples);
		free(noncartesian_data_nodcf);
		
		cartesian_data = (CPLX_T*) calloc (nuFFT_imp.Ntotal, sizeof(CPLX_T));
	}
	else if (nuFFT_direction == FORWARD) {
		cartesian_data = read_data_file(imagefilename); // possible: use Ndata=direct_nuDFT_imp.Ntotal
		noncartesian_data = (CPLX_T*) calloc (nuFFT_imp.Nsamples, sizeof(CPLX_T));
	}
	
	double t4 = timestamp();
	nuFFT_print_time(t4-t0);
	
	
	// transform data from source to target (using transform implementation)
	if (nuFFT_direction == ADJOINT)
		nuFFT_imp.adjoint(noncartesian_data, cartesian_data);
	else if (nuFFT_direction == FORWARD)
		nuFFT_imp.forward(cartesian_data, noncartesian_data);
	
	
	// write transformed (target) data to file
	if (nuFFT_direction == ADJOINT)
		write_data_file(imagefilename, cartesian_data, nuFFT_imp.Ntotal);
	else if (nuFFT_direction == FORWARD)
		write_data_file(kdatafilename, noncartesian_data, nuFFT_imp.Nsamples);
	
	
	nuFFT_imp.Free();
	
	free(noncartesian_data);
	free(cartesian_data);
	
	nuFFT_close();
	
	return 0;
}


nuFFT_implementation_t get_implementation(int argc, char* argv[])
{
	nuFFT_implementation_t nuFFT_imp;
	
	
	if (argc == 6) {
		
		char* impfilename = argv[5];
		nuFFT_print("\t");
		nuFFT_imp.Init(impfilename);
		
	}
	else if (argc >= 11) {
	
		int Ndim = atoi(argv[5]);
	
		int* imagesize = (int*) calloc (Ndim, sizeof(int));
	
		for (int dim=0; dim<Ndim; dim++)
			imagesize[dim] = atoi(argv[6+dim]);
			
		DATA_T maxaliasingerror = pow(10.0,atof(argv[6+Ndim]));
	
		DATA_T alpha = atof(argv[7+Ndim]);

		resampling_method_t resampling_method = get_resampling_method(argv[8+Ndim]);
		
		char* trajfilename = argv[9+Ndim];
    
		nuFFT_imp.Init(Ndim, imagesize, maxaliasingerror, alpha, resampling_method, trajfilename);
		
		free(imagesize);
		
		nuFFT_print("\t");
		nuFFT_imp.compute();
	}
	
	return nuFFT_imp;
}
