/* Michal Zarrouk, July 2013 */

#include "mex.h"
#include "nufftw.h"
#include <stdint.h>

char* get_mex_string(const mxArray* p)
{
	int nchar = mxGetNumberOfElements(p);
	char* str = (char*) calloc(nchar, sizeof(char));
	mxGetString(p, str, nchar+1);

	return str;
}


void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
/*	
	imp = nufftw(Nthreads, imagesize, maxerr, alphastart, alphaend, nalpha, resamplingmethod, tuning_heuristic, trajfilename);    (9)
		  nufftw(Nthreads, imagesize, maxerr, alphastart, alphaend, nalpha, resamplingmethod, tuning_heuristic, trajfilename, impfilename); (10)
	imp = nufftw(Nthreads, imagesize, maxerr, alphastart, alphaend, nalpha, resamplingmethod, tuning_heuristic, trajectory, sqrtdcf); (10)
		  nufftw(Nthreads, imagesize, maxerr, alphastart, alphaend, nalpha, resamplingmethod, tuning_heuristic, trajectory, sqrtdcf, impfilename); (11)
*/

	int Nthreads = mxGetScalar(prhs[0]);
	omp_set_num_threads(Nthreads);
	
	int Ndim = mxGetNumberOfElements(prhs[1]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = (int)((mxGetPr(prhs[1]))[dim]);

	DATA_T maxaliasingerror = mxGetScalar(prhs[2]);

	DATA_T alphastart = mxGetScalar(prhs[3]);
	DATA_T alphaend = mxGetScalar(prhs[4]);
	int Nalpha = mxGetScalar(prhs[5]);
	
	resampling_method_t resampling_method = get_resampling_method(get_mex_string(prhs[6]));

	tuning_heuristic_t tuning_heuristic = get_tuning_heuristic(get_mex_string(prhs[7]));
	
	
	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	
	
	nuFFT_tuner_t nuFFT_tuner (Ndim, imagesize, maxaliasingerror, alphastart, alphaend, Nalpha, resampling_method, fftw_flags);
	
	free(imagesize);
	
	
	if ((nlhs == 0 && nrhs == 10) || (nlhs == 1 && nrhs == 9)) {
	
		char* trajfilename = get_mex_string(prhs[8]);
		
		nuFFT_tuner.tune(tuning_heuristic, trajfilename, fftw_flags, Nthreads);
		
		free(trajfilename);
	}
	else if ((nlhs == 0 && nrhs == 11) || (nlhs == 1 && nrhs == 10)) {
	
		int Nsamples = mxGetNumberOfElements(prhs[8])/Ndim;
				
		TRAJ_T* trajectory = (TRAJ_T*) calloc (Nsamples*Ndim, sizeof(TRAJ_T));
		TRAJ_T* sqrtdcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
		for (int isample=0; isample<Nsamples; isample++) {
			for (int dim=0; dim<Ndim; dim++)
				trajectory[isample*Ndim+dim] = (mxGetPr(prhs[8]))[isample*Ndim+dim];
			sqrtdcf[isample] = (mxGetPr(prhs[9]))[isample];
		}
		
		nuFFT_tuner.tune(tuning_heuristic, Nsamples, trajectory, sqrtdcf, fftw_flags, Nthreads);
	}
	
	if (nlhs == 0) { // write implementation to file
	
		if (resampling_method == PRECOMPUTEDCONV && tuning_heuristic == FFTWTUNE)
			nuFFT_tuner.optimal_implementation.compute();

		char* impfilename;
		
		if (nrhs == 10)
			impfilename = get_mex_string(prhs[9]);
		else if (nrhs == 11)
			impfilename = get_mex_string(prhs[10]);
		
		nuFFT_tuner.optimal_implementation.write_impfile(impfilename);
		
		free(impfilename);
	}
	else if (nlhs == 1) {
		
		plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
		nuFFT_implementation_t* pnuFFT_imp = (nuFFT_implementation_t*) calloc(1, sizeof(nuFFT_implementation_t));
		*((uint64_t *)mxGetData(plhs[0])) = reinterpret_cast<uint64_t>(pnuFFT_imp);
		pnuFFT_imp = &nuFFT_tuner.optimal_implementation;
		
		nuFFT_tuner.optimal_implementation = NULL;
		nuFFT_tuner.trajectory = NULL;
		nuFFT_tuner.sqrtdcf = NULL;
	}
	

}
