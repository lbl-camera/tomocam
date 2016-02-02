/* Michal Zarrouk, June 2013 */

/* This file implements the forward and adjoint non-uniform fast Fourier transform (nuFFT). */

#include "nufft_gen.h"
#include "nufft_util.h"

#include "nufft_gridding.h"
#include "nufft_sparse.h"
#include "fftw3.h"


double nuFFT_init(resampling_method_t resampling_method)
{
	double t0 = timestamp();
	nuFFT_print("initializing external libraries: ");
	INIT_FFTW;
	if (resampling_method == PRECOMPUTEDCONV)
		oski_Init();
	double elapsed_time = timestamp() - t0;
	nuFFT_print_time(elapsed_time);
	return elapsed_time;
}

int nuFFT_close()
{
	nuFFT_print("cleaning external libraries\n");
	CLEAN_FFTW;
	oski_Close();
	
	return 0;
}

/* nuFFT_implementation_t - non-uniform fast Fourier transform implementation. */
struct nuFFT_implementation_t
{
	/* members */
	int Ndim; // number of dimensions
	int* N; // image size (number of pixels) in each dimension
	int Ntotal; // total number of pixels in image (Ntotal = prod(N))
	int Nsamples; // total number of nonuniform samples
	
	TRAJ_T* trajectory; // nonuniform sample locations
	TRAJ_T* sqrtdcf; // square root of nonuniform sample density comensation factors
	
	// nufft parameters
	DATA_T maxaliasingerror; // maximum aliasing error allowed for resampling (gridding)
	DATA_T alpha; // grid oversampling ratio
	
	// nufft data
	int* G; // oversampled grid size in each dimension
	int* Ncum; // cumulative number of pixels in image in each dimension (Ncum = cumprod(N))
	int* Gcum; // cumulative number of pixels in oversampled image in each dimension (Gcum = cumprod(G))
	int Gtotal; // total number of pixels in oversampled image (Gtotal = prod(G))
	int* nstart; // image FOV location within oversampled image in each dimension
	DATA_T** deapodization_factors; // deapodization factors in each dimension
	presampled_kernel_t* presampled_kernel_alldims; // presampled grid interpolation kernel in each dimension
	resampling_method_t resampling_method; // grid resampling method - on-the-fly, precomputed, etc.
	SpM_CSR_t SpM_CSR; // sparse gridding/convolution matrix

	CPLX_T* gridded_kdata; // gridded, oversampled (k-space) data
	CPLX_T* oversampled_image; // Fourier transform of gridded oversampled k-space data
#if DATA_TYPE == DOUBLE
	fftw_plan fftw_forward_plan; // forward fftw plan, transforms from oversampled_image to gridded_kdata
	fftw_plan fftw_adjoint_plan; // adjoint fftw plan, transforms from gridded_kdata to oversampled_image
#elif DATA_TYPE == FLOAT
	fftwf_plan fftw_forward_plan; // forward fftw plan, transforms from oversampled_image to gridded_kdata
	fftwf_plan fftw_adjoint_plan; // adjoint fftw plan, transforms from gridded_kdata to oversampled_image
#endif

	bool free_trajectory;
	
		
	/* methods */
	nuFFT_implementation_t ()
	{ set_defaults(); }
	
	nuFFT_implementation_t (int ndim, int* imagesize, DATA_T epsmax, DATA_T a, resampling_method_t res_method, char* trajfilename)
	{ Init (ndim, imagesize, epsmax, a, res_method, trajfilename); }
		
	nuFFT_implementation_t (int ndim, int* imagesize, DATA_T epsmax, DATA_T a, resampling_method_t res_method, int nsamples, TRAJ_T* traj, TRAJ_T* sqrtdenscomp)
	{ Init (ndim, imagesize, epsmax, a, res_method, nsamples, traj, sqrtdenscomp); }

	nuFFT_implementation_t (char* impfilename)
	{ Init (impfilename); }
	
	
	int set_defaults()
	{
		N = NULL;
		G = NULL;
		Ncum = NULL;
		Gcum = NULL;
		nstart = NULL;
		deapodization_factors = NULL;
		presampled_kernel_alldims = NULL;
		gridded_kdata = NULL;
		oversampled_image = NULL;
		fftw_forward_plan = NULL;
		fftw_adjoint_plan = NULL;
		trajectory = NULL;
		sqrtdcf = NULL;
		
		free_trajectory = false;
		
		return 0;
	}
	
	
	
	int set_params(int ndim, int* imagesize, DATA_T epsmax, DATA_T a, resampling_method_t res_method)
	{
		set_defaults();
		
		Ndim = ndim;
		N = (int*) calloc (Ndim, sizeof(int));
		for (int dim=0; dim<Ndim; dim++)		
			N[dim] = imagesize[dim];
		maxaliasingerror = epsmax;
		alpha = a;
			
		resampling_method = res_method;
		
		calc_params();
		
		
		return 0;
	}
	
	int calc_params()
	{
		G      = (int*) calloc (Ndim,   sizeof(int));
		Ncum   = (int*) calloc (Ndim+1, sizeof(int));
		Gcum   = (int*) calloc (Ndim+1, sizeof(int));
		nstart = (int*) calloc (Ndim,   sizeof(int));
		Ncum[0] = 1;
		Gcum[0] = 1;
		for (int dim=0; dim<Ndim; dim++) {
			
			G[dim] = OversampledImageSize(alpha, N[dim]);
			
			nstart[dim] = ceil((G[dim]-N[dim])/2.);
			
			Ncum[dim+1] = Ncum[dim]*N[dim];
			Gcum[dim+1] = Gcum[dim]*G[dim];
		}
		Ntotal = Ncum[Ndim];
		Gtotal = Gcum[Ndim];
		
		return 0;
	}
	
	
	int Init (int ndim, int* imagesize, DATA_T epsmax, DATA_T a, resampling_method_t res_method, char* trajfilename)
	{
		set_params(ndim, imagesize, epsmax, a, res_method);
		
		read_trajectory_file (trajfilename, Ndim, trajectory, sqrtdcf, Nsamples);
		free_trajectory = true;
		
		return 0;
	}
	
	
	int Init (int ndim, int* imagesize, DATA_T epsmax, DATA_T a, resampling_method_t res_method, int nsamples, TRAJ_T* traj, TRAJ_T* sqrtdenscomp)
	{
		set_params(ndim, imagesize, epsmax, a, res_method);
				
		Nsamples = nsamples;
		trajectory = traj;
		sqrtdcf = sqrtdenscomp;
		
		free_trajectory = false;
		
		return 0;
	}
	
	
	double Init (char* impfilename)
	{
		double t0 = timestamp();
		nuFFT_print("loading implementation from file: ");	
	
		FILE* impfile = nufft_fopen(impfilename,"rb");
		fread(&Ndim, sizeof(int), 1, impfile);
		N = (int*) calloc (Ndim, sizeof(int));
		fread(N, sizeof(int), Ndim, impfile);
		fread(&maxaliasingerror, sizeof(DATA_T), 1, impfile);
		fread(&alpha, sizeof(DATA_T), 1, impfile);
		fread(&resampling_method, sizeof(resampling_method_t), 1, impfile);
		
		calc_params();
		
		if (resampling_method == PRECOMPUTEDCONV) {
			calc_deap_and_kernel(false);
			SpM_CSR.read_from_file(impfile);
			Nsamples = SpM_CSR.Ncols;
		}
		else if (resampling_method == ONTHEFLYCONV) {
			calc_deap_and_kernel();
			fread(&Nsamples, sizeof(int), 1, impfile);
			trajectory = (TRAJ_T*) calloc (Nsamples*Ndim, sizeof(TRAJ_T));
			fread(trajectory, sizeof(TRAJ_T), Nsamples*Ndim, impfile);
		}
		sqrtdcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
		fread(sqrtdcf, sizeof(TRAJ_T), Nsamples, impfile);
		fclose(impfile);
		
		double elapsed_time = timestamp() - t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}
	
	
	double write_impfile(char* impfilename)
	{
		double t0 = timestamp();
		nuFFT_print("writing implementation to file: ");
	
		FILE* impfile = nufft_fopen(impfilename,"wb");
		fwrite(&Ndim, sizeof(int), 1, impfile);
		fwrite(N, Ndim, sizeof(int), impfile);
		fwrite(&maxaliasingerror, sizeof(DATA_T), 1, impfile);
		fwrite(&alpha, sizeof(DATA_T), 1, impfile);
		fwrite(&resampling_method, sizeof(resampling_method_t), 1, impfile);
		
		if (resampling_method == PRECOMPUTEDCONV) {
			SpM_CSR.write_to_file(impfile);
		}
		else if (resampling_method == ONTHEFLYCONV) {
			fwrite(&Nsamples, sizeof(int), 1, impfile);
			fwrite(trajectory, sizeof(TRAJ_T), Nsamples*Ndim, impfile);
		}
		fwrite(sqrtdcf, sizeof(TRAJ_T), Nsamples, impfile);
		
		fclose(impfile);
		
		double elapsed_time = timestamp() - t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}
	
	double compute()
	{
		return compute(resampling_method);
	}
	
	double compute(resampling_method_t resampling_method)
	{
		double t0 = timestamp();
		nuFFT_print("computing implementation data: ");

		calc_deap_and_kernel();
		if (resampling_method == PRECOMPUTEDCONV) 
			calc_spm();
		
		double elapsed_time = timestamp()-t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}	
	
	
	int calc_deap_and_kernel(bool calckernel = true)
	{
		
		deapodization_factors = (DATA_T**) calloc (Ndim, sizeof(DATA_T*));
		
		if (calckernel)
			presampled_kernel_alldims = (presampled_kernel_t*) calloc (Ndim, sizeof(presampled_kernel_t));
		
		#pragma omp parallel for
		for (int dim=0; dim<Ndim; dim++) {
			DATA_T beta;
			DATA_T W = KernelWidth (maxaliasingerror, alpha, N[dim], &beta);
			//this loop can be optimized by each iteration telling the others where to look
			
			// calculate deapodization factors
			deapodization_factors[dim] = (DATA_T*) calloc (N[dim], sizeof(DATA_T));
			int x = -floor(N[dim]/2);
			DATA_T S = KernelSamplingRatio(maxaliasingerror, alpha, LINEAR);
			int SG = ceil(S*G[dim]);
			for (int i=0; i<N[dim]; i++, x++) {

				// correction factor for linear interpolation from a presampled gridding kernel
				DATA_T arg = (DATA_T)x/SG*M_PI;
				arg = pow(sin(arg)/arg,2);
				if (x==0)
					arg = 1;

				deapodization_factors[dim][i] = KaiserBessel_ImageDomain ((DATA_T)x, W, G[dim], beta) * G[dim] * arg;
				
				// +-1 modulation (instead of fftshifting after fft)
				if (i % 2 == 1)
					deapodization_factors[dim][i] *= -1;

			}
			
			if (calckernel) // presample kernel
				presampled_kernel_alldims[dim].Init(N[dim], W, beta, alpha, maxaliasingerror, G[dim]);
	
		}
		
		return 0;
	}
	

	int calc_spm()
	{
		SpM_COO_t SpM_COO = nuFFT_precompute_COO (trajectory, sqrtdcf, Ndim, Nsamples, presampled_kernel_alldims, G, Gtotal);
		SpM_CSR.COO2CSR(SpM_COO);
		
		return 0;
	}
	
	
	double tune(unsigned fftwflags, int npartitions)
	{
		return tune(fftwflags, npartitions, resampling_method);
	}
	
	double tune(unsigned fftwflags, int npartitions, resampling_method_t resampling_method)
	{
		double t0 = timestamp();
		nuFFT_print("tuning: ");
	
		tune_fftw(fftwflags);
		

		if (resampling_method == PRECOMPUTEDCONV)
			SpM_CSR.tune(npartitions, gridded_kdata);
		
		
		double elapsed_time = timestamp() - t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}
	
	int tune_fftw(unsigned fftwflags)
	{
		// plan fftw transforms
		gridded_kdata     = (CPLX_T*) calloc (Gcum[Ndim], sizeof(CPLX_T));
		oversampled_image = (CPLX_T*) calloc (Gcum[Ndim], sizeof(CPLX_T));
		
	#if DATA_TYPE == DOUBLE
		fftw_forward_plan = fftw_plan_dft(Ndim,  G, oversampled_image, gridded_kdata,     FFTW_FORWARD,  fftwflags);
		fftw_adjoint_plan = fftw_plan_dft(Ndim,  G, gridded_kdata,     oversampled_image, FFTW_BACKWARD, fftwflags);		
	#elif DATA_TYPE == FLOAT
		fftw_forward_plan = fftwf_plan_dft(Ndim, G, oversampled_image, gridded_kdata,     FFTW_FORWARD,  fftwflags);
		fftw_adjoint_plan = fftwf_plan_dft(Ndim, G, gridded_kdata,     oversampled_image, FFTW_BACKWARD, fftwflags);		
	#endif
		
		return 0;
	}
	// destructor:
	int Free()
	{
		nuFFT_print("deleting nuFFT implementation\n");
		
		SpM_CSR.Free();
		free(N);
		free(G);
		free(Ncum);
		free(Gcum);
		free(nstart);
		if (presampled_kernel_alldims != NULL)
			for (int dim=0; dim<Ndim; dim++)
				presampled_kernel_alldims[dim].~presampled_kernel_t();
		free(presampled_kernel_alldims);
		if (deapodization_factors != NULL)
			for (int dim=0; dim<Ndim; dim++)
				free(deapodization_factors[dim]);
		free(deapodization_factors);
	#if DATA_TYPE == DOUBLE
		fftw_destroy_plan(fftw_forward_plan);
		fftw_destroy_plan(fftw_adjoint_plan);
	#elif DATA_TYPE == FLOAT
		fftwf_destroy_plan(fftw_forward_plan);
		fftwf_destroy_plan(fftw_adjoint_plan);
	#endif
		free(gridded_kdata);
		free(oversampled_image);
		
		if (free_trajectory) {
			free(trajectory);
			free(sqrtdcf);
		}
		return 0;
	}
	
	
	// nuFFT_implementation_t::forward - forward nuFFT, from uniform image to non-uniform k-space data
	double forward (CPLX_T* cartesian_data, CPLX_T* noncartesian_data)
	{
		double t0 = timestamp();	
		nuFFT_print("forward transformation: ");
		
		// zero pad (oversample in image space) and deapodize (compensate for gridding convolution)
		deap(FORWARD, cartesian_data);

		// Fourier transform oversampled image to k-space
		fftw(FORWARD);
		
		// convolve the uniform k-space data onto nonuniform locations
		grid(FORWARD, noncartesian_data);
		
		double elapsed_time = timestamp()-t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}
	
	
	// nuFFT_implementation_t::adjoint - adjoint nuFFT, from non-uniform k-space data to uniform image
	double adjoint (CPLX_T* noncartesian_data, CPLX_T* cartesian_data)
	{
		double t0 = timestamp();	
		nuFFT_print("adjoint transformation: ");
		
		// grid the nonuniform k-space data onto a uniform grid
		grid(ADJOINT, noncartesian_data);
				
		// Fourier transform the k-space grid onto image space
		fftw(ADJOINT);
		
		// truncate FOV and deapodize (compensate for gridding convolution)
		deap(ADJOINT, cartesian_data);
		
		double elapsed_time = timestamp()-t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}


	// nuFFT_implementation_t::nuFFT_grid - gridding (convolution) phase of the nuFFT
	double grid (nuDFT_direction_t nuFFT_direction, CPLX_T*& noncartesian_data)
	{
		//maybe the timing should be out here?
		if (resampling_method == ONTHEFLYCONV)
			return OnTheFlyConv (nuFFT_direction, noncartesian_data);
			
		else if (resampling_method == PRECOMPUTEDCONV)
			return PrecomputedConv (nuFFT_direction, noncartesian_data);
	}
	
	
	double OnTheFlyConv (nuDFT_direction_t nuFFT_direction, CPLX_T*& noncartesian_data)
	{
		double t0 = timestamp();	
	
		#pragma omp parallel for
		for (int isample=0; isample<Nsamples; isample++) {
			sample_gridding_info_t sgi(Ndim, &trajectory[isample*Ndim], sqrtdcf[isample], presampled_kernel_alldims, G);
		
			// find matrix element number from row, column etc indices for each element.
			for (int r=0; r<sgi.nelemcum[Ndim]; r++) {
				int igrid;
				DATA_T value;
				
				sgi.compute_grid_point(r, G, igrid, value);

				if (nuFFT_direction == ADJOINT) {
					#pragma omp atomic
					((DATA_T*)gridded_kdata)[igrid*2]   += value * ((DATA_T*)noncartesian_data)[isample*2];
					#pragma omp atomic
					((DATA_T*)gridded_kdata)[igrid*2+1] += value * ((DATA_T*)noncartesian_data)[isample*2+1];
				}
				else if (nuFFT_direction == FORWARD) {
					#pragma omp atomic
					((DATA_T*)noncartesian_data)[isample*2]   += value * ((DATA_T*)gridded_kdata)[igrid*2];
					#pragma omp atomic
					((DATA_T*)noncartesian_data)[isample*2+1] += value * ((DATA_T*)gridded_kdata)[igrid*2+1];
				}
			}
		}
	
		return timestamp()-t0;
	}


	double PrecomputedConv (nuDFT_direction_t nuFFT_direction, CPLX_T*& noncartesian_data)
	{	
		double t0 = timestamp();	

		if (nuFFT_direction == ADJOINT) {
			oski_vecview_t x = oski_CreateMultiVecView ((DATA_T*)noncartesian_data, SpM_CSR.Ncols, 2*SpM_CSR.nch, LAYOUT_ROWMAJ, 2*SpM_CSR.nch);
			#pragma omp parallel for
			for (int ipartition=0; ipartition<SpM_CSR.npartitions; ipartition++)
				oski_MatMult (SpM_CSR.pGamma[ipartition], OP_NORMAL, 1.0f, x, 0.0f, SpM_CSR.py[ipartition]);
			oski_DestroyVecView(x);
		}
		else if (nuFFT_direction == FORWARD) {
			oski_vecview_t y = oski_CreateMultiVecView ((DATA_T*)noncartesian_data, SpM_CSR.Ncols, 2*SpM_CSR.nch, LAYOUT_ROWMAJ, 2*SpM_CSR.nch);
			oski_MatMult (SpM_CSR.Gamma, OP_TRANS, 1.0f, SpM_CSR.x, 0.0f, y);
			oski_DestroyVecView(y);
		}
		return timestamp()-t0;
	}


	double fftw (nuDFT_direction_t nuDFT_direction)
	{
		double t0 = timestamp();
		if (nuDFT_direction == ADJOINT)
	#if DATA_TYPE == DOUBLE
			fftw_execute(fftw_adjoint_plan);
	#elif DATA_TYPE == FLOAT
			fftwf_execute(fftw_adjoint_plan);
	#endif
		else if (nuDFT_direction == FORWARD)
	#if DATA_TYPE == DOUBLE
			fftw_execute(fftw_forward_plan);
	#elif DATA_TYPE == FLOAT
			fftwf_execute(fftw_forward_plan);
	#endif
	
		return timestamp() - t0;
	}
	
	
	double deap (nuDFT_direction_t nuDFT_direction, CPLX_T*& cartesian_data)
	{
	
		double t0 = timestamp();
		
		if (nuDFT_direction == ADJOINT) {
			#pragma omp parallel for
			for (int r=0; r<Ntotal; r++) {
				int igrid = 0;
				DATA_T value = 1;
				int q = r;
				for (int dim=Ndim-1; dim>=0; dim--) {
					int s = floor((DATA_T)q/Ncum[dim]);
					value *= deapodization_factors[dim][s];
					int l = nstart[dim]+s;
					igrid = l+G[dim]*igrid;
					q = q % Ncum[dim];
				}
	
				((DATA_T*)cartesian_data)[r*2]   = ((DATA_T*)oversampled_image)[igrid*2]   / value;
				((DATA_T*)cartesian_data)[r*2+1] = ((DATA_T*)oversampled_image)[igrid*2+1] / value;
			}
		}
		else if (nuDFT_direction == FORWARD) {
			#pragma omp parallel for
			for (int igrid=0; igrid<Gtotal; igrid++) {
				int r = 0;
				DATA_T value = 1;
				int q = igrid;
				int lsum = 0;
				bool isinfov = true;
				int dim = Ndim-1;
				while (isinfov & dim>=0) {
					int s = floor((DATA_T)q/Gcum[dim]);
					int l = s-nstart[dim];
			
					if (l < 0 | l >= N[dim])
						isinfov = false;
					else {
						value *= deapodization_factors[dim][l];
						r = l+N[dim]*r;
						q = q % Gcum[dim];
					}
					dim--;
				}
		
				if (isinfov) {
					((DATA_T*)oversampled_image)[igrid*2]   = ((DATA_T*)cartesian_data)[r*2]   / value;
					((DATA_T*)oversampled_image)[igrid*2+1] = ((DATA_T*)cartesian_data)[r*2+1] / value;
				}
				else {
					((DATA_T*)oversampled_image)[igrid*2]   = 0;
					((DATA_T*)oversampled_image)[igrid*2+1] = 0;
				}
			}
		
		}
		return timestamp() - t0;
	}
	
};
