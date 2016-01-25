/* Michal Zarrouk, July 2013 */

#include "nufftw.h"

int main(int argc, char* argv[])
{
	/* usage:
		nufftw Nthreads Ndim imagesize maxerr_power alphastart alphaend nalpha resamplingmethod tuning_heuristic trajfilename
		nufftw Nthreads Ndim imagesize maxerr_power alphastart alphaend nalpha resamplingmethod tuning_heuristic trajfilename impfilename
	*/

	// get input data
	nuFFT_print("** nuFFTW **\n");
	
	int Nthreads = atoi(argv[1]);
	omp_set_num_threads(Nthreads);
    
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
    
	DATA_T maxaliasingerror = pow(10.0,atof(argv[3+Ndim]));
	
	DATA_T alphastart = strtod(argv[4+Ndim], NULL);
	DATA_T alphaend = strtod(argv[5+Ndim], NULL);
	int Nalpha = atoi(argv[6+Ndim]);
	
	resampling_method_t resampling_method = get_resampling_method(argv[7+Ndim]);
	
	tuning_heuristic_t tuning_heuristic = get_tuning_heuristic(argv[8+Ndim]);
	
	char* trajfilename = argv[9+Ndim];



	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
		
	//how about generating fftw wisdom for N*alpha...
	nuFFT_tuner_t nuFFT_tuner (Ndim, imagesize, maxaliasingerror, alphastart, alphaend, Nalpha, resampling_method, fftw_flags);
	
	nuFFT_tuner.tune(tuning_heuristic, trajfilename, fftw_flags, Nthreads);
	
	// write implementation to file
	if (argc == 11+Ndim) {
		
		if (resampling_method == PRECOMPUTEDCONV && tuning_heuristic == FFTWTUNE)
			nuFFT_tuner.optimal_implementation.compute();
		
		char* impfilename = argv[10+Ndim];
		
		nuFFT_tuner.optimal_implementation.write_impfile(impfilename);
	}
    
    
    free(imagesize);
    
    return 0;
}
