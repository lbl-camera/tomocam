/* Michal Zarrouk, July 2013 */

#include "mex.h"
#include "nufft.h"
#include <stdint.h>

char* get_mex_string(const mxArray* p)
{
	int nchar = mxGetNumberOfElements(p);
	char* str = (char*) calloc(nchar, sizeof(char));
	mxGetString(p, str, nchar+1);

	return str;
}

bool nuFFT_mex_first_call = true;

void nuFFT_mex_close(void)
{
	nuFFT_close();
	nuFFT_mex_first_call = true;
}


CPLX_T* complex2vec(const mxArray *pm, int N)
{
	DATA_T* data_real = (DATA_T*) mxGetData(pm);
	DATA_T* data_imag = (DATA_T*) mxGetImagData(pm);
	
	CPLX_T* data = (CPLX_T*) calloc (N, sizeof(CPLX_T));
	for (int i=0; i<N; i++) {
		((DATA_T*)data)[i*2]   = data_real[i];
		((DATA_T*)data)[i*2+1] = data_imag[i];
	}
	
	
	return data;

}


mxArray* vec2complex(CPLX_T* data, int Ndim, int* N)
{

	mxArray* pm = mxCreateNumericArray(Ndim, N, MEXDATA_CLASS, mxCOMPLEX);
	
	DATA_T* data_real = (DATA_T*) mxGetData(pm);
	DATA_T* data_imag = (DATA_T*) mxGetImagData(pm);
	
	int Ntotal=1;
	for (int dim=0; dim<Ndim; dim++)
		Ntotal *= N[dim];
	
	for (int i=0; i<Ntotal; i++) {
		data_real[i] = ((DATA_T*)data)[i*2];
		data_imag[i] = ((DATA_T*)data)[i*2+1];
	}
	
	return pm;

}


void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	if (nuFFT_mex_first_call) {
		nuFFT_mex_first_call = false;
		mexAtExit(nuFFT_mex_close);
	}


	char* funname = get_mex_string(prhs[0]);
	
	if (strcmp(funname,"create_impfile") == 0) {
		/*	nufft_mex('create_nufft_impfile', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajectory, sqrtdcf, impfilename);
			nufft_mex('create_nufft_impfile', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajfilename, impfilename);
			nufft_mex('create_nufft_impfile', imp, impfilename);
		*/
		
		nuFFT_print("** create nuFFT implementation file**\n");
	
		char* impfilename;
		
		if (nrhs == 9 || nrhs == 8) {
			
			// nufft_mex('create_nufft_impfile', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajectory, sqrtdcf, impfilename);
			// nufft_mex('create_nufft_impfile', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajfilename, impfilename);
			
			int Nthreads = mxGetScalar(prhs[1]);
			omp_set_num_threads(Nthreads);
	
			int Ndim = mxGetNumberOfElements(prhs[2]);
	
			int* imagesize = (int*) calloc (Ndim, sizeof(int));
			for (int dim=0; dim<Ndim; dim++)
				imagesize[dim] = (int)((mxGetPr(prhs[2]))[dim]);
		
			DATA_T maxaliasingerror = (DATA_T)mxGetScalar(prhs[3]);
	
			DATA_T alpha = (DATA_T)mxGetScalar(prhs[4]);

			resampling_method_t resampling_method = get_resampling_method(get_mex_string(prhs[5]));
			
			nuFFT_implementation_t nuFFT_imp;
			
			if (nrhs == 9) {
	
				TRAJ_T* trajectory = (TRAJ_T*) mxGetData(prhs[6]);

				int Nsamples = mxGetNumberOfElements(prhs[6])/Ndim;
	
				TRAJ_T* sqrtdcf = (TRAJ_T*) mxGetData(prhs[7]);
	
				impfilename = get_mex_string(prhs[8]);
	
				nuFFT_imp.Init(Ndim, imagesize, maxaliasingerror, alpha, resampling_method, Nsamples, trajectory, sqrtdcf);
			}
			else if (nrhs == 8) {
				char* trajfilename = get_mex_string(prhs[6]);
				
				impfilename = get_mex_string(prhs[7]);
	
				nuFFT_imp.Init(Ndim, imagesize, maxaliasingerror, alpha, resampling_method, trajfilename);
				free(trajfilename);
			
			}
			free(imagesize);
			
			nuFFT_imp.compute();
	
			nuFFT_imp.write_impfile(impfilename);

			nuFFT_imp.Free();
	
		}
		else if (nrhs == 3) {
			
			// nufft_mex('create_nufft_impfile', imp, impfilename);
			
			nuFFT_implementation_t nuFFT_imp = *(reinterpret_cast<nuFFT_implementation_t*>(*((uint64_t *)mxGetData(prhs[1]))));
			impfilename = get_mex_string(prhs[2]);
	
			nuFFT_imp.write_impfile(impfilename);
			
		}
		
		free(impfilename);
	}
	
	else if (strcmp(funname,"init") == 0) {

		/*	imp = nufft_mex('init', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajectory, sqrtdcf);
			imp = nufft_mex('init', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajfilename);
			imp = nufft_mex('init', Nthreads, impfilename);
		*/
		
		nuFFT_print("** create nuFFT implementation **\n");
	
		plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
		nuFFT_implementation_t* pnuFFT_imp = (nuFFT_implementation_t*) calloc(1, sizeof(nuFFT_implementation_t));
		*((uint64_t *)mxGetData(plhs[0])) = reinterpret_cast<uint64_t>(pnuFFT_imp);
		// no need to mexMakeMemoryPersistent after all, because I didn't have to mxCalloc pnuFFT_imp.

		int Nthreads = mxGetScalar(prhs[1]);
		omp_set_num_threads(Nthreads);

		if (nrhs == 8 || nrhs == 7) {
			
			// imp = nufft_mex('init', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajectory, sqrtdcf);
			// imp = nufft_mex('init', Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajfilename);
			
			int Ndim = mxGetNumberOfElements(prhs[2]);
	
			int* imagesize = (int*) calloc (Ndim, sizeof(int));
			for (int dim=0; dim<Ndim; dim++)
				imagesize[dim] = (int)((mxGetPr(prhs[2]))[dim]);
		
			DATA_T maxaliasingerror = (DATA_T)mxGetScalar(prhs[3]);
	
			DATA_T alpha = (DATA_T)mxGetScalar(prhs[4]);

			resampling_method_t resampling_method = get_resampling_method(get_mex_string(prhs[5]));
	
			if (nrhs == 8) {
				
				pnuFFT_imp->set_params(Ndim, imagesize, maxaliasingerror, alpha, resampling_method);
				
				pnuFFT_imp->Nsamples = mxGetNumberOfElements(prhs[6])/Ndim;
				pnuFFT_imp->trajectory = (TRAJ_T*) calloc (pnuFFT_imp->Nsamples*Ndim, sizeof(TRAJ_T));
				pnuFFT_imp->sqrtdcf = (TRAJ_T*) calloc (pnuFFT_imp->Nsamples, sizeof(TRAJ_T));
				for (int isample=0; isample<pnuFFT_imp->Nsamples; isample++) {
					for (int dim=0; dim<Ndim; dim++)
						pnuFFT_imp->trajectory[isample*Ndim+dim] = ((TRAJ_T*)mxGetData(prhs[6]))[isample*Ndim+dim];
					pnuFFT_imp->sqrtdcf[isample] = ((TRAJ_T*)mxGetData(prhs[7]))[isample];
				}			
				pnuFFT_imp->free_trajectory = true;
			}
			else if (nrhs == 7) {

				char* trajfilename = get_mex_string(prhs[6]);
				
				pnuFFT_imp->Init(Ndim, imagesize, maxaliasingerror, alpha, resampling_method, trajfilename);
				
				free(trajfilename);
			}
			
			free(imagesize);
		
			pnuFFT_imp->compute();
			
		}
		else if (nrhs == 3) {
		
			// imp = nufft_mex('init', Nthreads, impfilename);
			
			char* impfilename = get_mex_string(prhs[2]);
			
			pnuFFT_imp->Init(impfilename);
			
			free(impfilename);	
		}
		
		nuFFT_init(pnuFFT_imp->resampling_method);

		unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
		pnuFFT_imp->tune(fftw_flags, Nthreads);
		
	}
	else if (strcmp(funname,"forward") == 0) {
		
		// kdata = nufft_mex('forward_nufft', Nthreads, imp, image_data);
		
		nuFFT_print("** forward nuFFT **\n");
		
		int Nthreads = mxGetScalar(prhs[1]);
		omp_set_num_threads(Nthreads);

		nuFFT_implementation_t nuFFT_imp = *(reinterpret_cast<nuFFT_implementation_t*>(*((uint64_t *)mxGetData(prhs[2]))));
		
		CPLX_T* cartesian_data = complex2vec(prhs[3], nuFFT_imp.Ntotal);
				
		CPLX_T* noncartesian_data = (CPLX_T*) calloc (nuFFT_imp.Nsamples, sizeof(CPLX_T));
		nuFFT_imp.forward(cartesian_data, noncartesian_data);
		
		free(cartesian_data);
		
		int datasize[] = {nuFFT_imp.Nsamples, 1};
		plhs[0] = vec2complex(noncartesian_data, 2, datasize);
		free(noncartesian_data);
	
	}
	else if (strcmp(funname,"adjoint") == 0) {
		
		// image_data = nufft_mex('adjoint_nufft', Nthreads, imp, kdata_denscomp);
		
		nuFFT_print("** adjoint nuFFT **\n");
		
		int Nthreads = mxGetScalar(prhs[1]);
		omp_set_num_threads(Nthreads);

		nuFFT_implementation_t nuFFT_imp = *(reinterpret_cast<nuFFT_implementation_t*>(*((uint64_t *)mxGetData(prhs[2]))));
		
		CPLX_T* noncartesian_data = complex2vec(prhs[3], nuFFT_imp.Nsamples);

		CPLX_T* cartesian_data = (CPLX_T*) calloc (nuFFT_imp.Ntotal, sizeof(CPLX_T));
		nuFFT_imp.adjoint(noncartesian_data, cartesian_data);

		free(noncartesian_data);
		
		
		plhs[0] = vec2complex(cartesian_data, nuFFT_imp.Ndim, nuFFT_imp.N);
		free(cartesian_data);

	}
	
	else if (strcmp(funname,"delete") == 0) {
		
		nuFFT_implementation_t nuFFT_imp = *(reinterpret_cast<nuFFT_implementation_t*>(*((uint64_t *)mxGetData(prhs[1]))));
		nuFFT_imp.Free();
	}
}



