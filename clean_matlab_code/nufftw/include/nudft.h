/* Michal Zarrouk, June 2013 */

/* This file implements the forward and adjoint non-uniform discrete Fourier transform,
	aka direct/slow/exact non-uniform Fourier transform - nuDFT. */

#include "nufft_gen.h"

/* nuDFT_implementation_t - non-uniform discrete Fourier transform implementation. */
struct nuDFT_implementation_t
{
	/* members */
	int Ndim;     // Number of dimensions of transform
	int* N;       // uniform grid size, an array of Ndim elements
	int Ntotal;   // total number of grid samples (Ntotal = prod(N))
	int Nsamples; // number of nonuniform samples
	TRAJ_T* trajectory; // nonuniform sample coordinates
	TRAJ_T* sqrtdcf;  // square root of density compensation factors for nonuniform samples
	
	
	/* methods */
	// constructor:
	nuDFT_implementation_t (char* trajfilename, int ndim, int* n)
	{ Init(trajfilename, ndim, n); }
	
	// nuDFT_implementation_t::Init - initializes the nuDFT with information about the trajectory and image size.
	int Init (char* trajfilename, int ndim, int* n)
	{
		Ndim = ndim;
		read_trajectory_file(trajfilename, Ndim, trajectory, sqrtdcf, Nsamples);

		N = (int*) calloc (Ndim, sizeof(int));
		Ntotal = 1;
		for (int dim=0; dim<Ndim; dim++)
		{
			N[dim] = n[dim];
			Ntotal *= N[dim];
		}
		
		return 0;
	}
	
	// destructor:
	~nuDFT_implementation_t()
	{
		free(N);
		free(trajectory);
		free(sqrtdcf);
	}
	
	
	// nuDFT_implementation_t::forward - forward nuDFT, from uniform image to nonuniform k-space data
	double forward (CPLX_T* cartesian_data, CPLX_T* noncartesian_data)
	{	
		double t0 = timestamp();
		nuFFT_print("forward transformation: ");
		
		/* This is essentially matrix vector multiplication (in parallel):
			for each row (pixel in image), we caculate the effect of all of the samples on it. */
		#pragma omp parallel for
		for (int igrid=0; igrid<Ntotal; igrid++) {
		
			/* The intuitive way to calculate each pixel in the image would be to nest Ndim loops,
				but that isn't robust for any number of dimensions.
				Instead, we have one loop that runs on all image pixels,
				and we caluclate for each pixel it's indices within the image, similar to MATLAB's ind2sub function. */
			int* gridindices = num_to_matrix_indices (Ndim, N, igrid);
	
			// discrete integration - run on samples and accumulate effects.
			for (int isample=0; isample<Nsamples; isample++) {
			
				// each sample contributes phase in all dimensions (kx*x + ky*y + kz*z + ...)
				DATA_T arg = 0;
				for (int dim=0; dim<Ndim; dim++)
					arg += trajectory[isample*Ndim+dim] * (gridindices[dim]-floor(N[dim]/2));
				DATA_T phase = -2*M_PI*arg;
	
				// Fourier integration
				noncartesian_data[isample][0] += cartesian_data[igrid][0] * cos(phase) - cartesian_data[igrid][1] * sin(phase);
				noncartesian_data[isample][1] += cartesian_data[igrid][0] * sin(phase) + cartesian_data[igrid][1] * cos(phase);
				
			}
			
			free(gridindices);
			
		}
		
		double elapsed_time = timestamp() - t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
	}
	
	
	// nuDFT_implementation_t::adjoint - adjoint nuDFT, from nonuniform k-space data to uniform image
	double adjoint (CPLX_T* noncartesian_data, CPLX_T* cartesian_data)
	{
		double t0 = timestamp();
		nuFFT_print("adjoint transformation: ");
		
		/* This is essentially matrix vector multiplication (in parallel):
			for each row (pixel in image), we caculate the effect of all of the samples on it. */
		#pragma omp parallel for
		for (int igrid=0; igrid<Ntotal; igrid++) {
			
			/* The intuitive way to calculate each pixel in the image would be to nest Ndim loops,
				but that isn't robust for any number of dimensions.
				Instead, we have one loop that runs on all image pixels,
				and we caluclate for each pixel it's indices within the image, similar to MATLAB's ind2sub function. */
			int* gridindices = num_to_matrix_indices (Ndim, N, igrid);
	
			// run on samples and accumulate effects - basically a discrete integraion.
			// maybe use memset instead?
			for (int isample=0; isample<Nsamples; isample++) {
				// each sample conributes phase in all dimensions (kx*x + ky*y + kz*z + ...)
				DATA_T arg = 0;
				for (int dim=0; dim<Ndim; dim++)
					arg += trajectory[isample*Ndim+dim] * (gridindices[dim]-floor(N[dim]/2));
				DATA_T phase = 2*M_PI*arg;
		
				// Fourier integration
				cartesian_data[igrid][0] += noncartesian_data[isample][0] * cos(phase) - noncartesian_data[isample][1] * sin(phase);
				cartesian_data[igrid][1] += noncartesian_data[isample][0] * sin(phase) + noncartesian_data[isample][1] * cos(phase);
				
			}
			
			free(gridindices);
			
		}
		
		double elapsed_time = timestamp() - t0;
		nuFFT_print_time(elapsed_time);
		return elapsed_time;
		
		
		/* The following code does the same as the above but uses a nice digital counter for grid locations,
			but that inhibits parallelization of the code so I prefered to ditch the counter.
			maybe we should check whether parallelization is enabled and if not then do this. */
		/*
		int* gridindices = (int*) calloc (sampled_data.Ndim, sizeof(int));
		gridindices[0] = -1;
		for (int dim=1; dim<sampled_data.Ndim; dim++)
			gridindices[dim] = 0;

		for (int igrid=0; igrid<gridded_kdata.Ntotal; igrid++)
		{
			//this is like a digital counter
			int dim = 0;
			gridindices[dim]++;
			while (gridindices[dim]>=gridded_kdata.imagesize[dim])
			{
				gridindices[dim] = 0;
				dim++;
				gridindices[dim]++;
			}
	
			gridded_kdata.image[igrid] = 0;
	
			for (int isample=0; isample<sampled_data.Nsamples; isample++)
			{
				// each sample conributes phase in all dimensions (kx*x + ky*y + kz*z + ...)
				DATA_T arg = 0;
				for (int dim=0; dim<sampled_data.Ndim; dim++)
					arg += sampled_data.kcoord[isample][dim] * (gridindices[dim]-floor(gridded_kdata.imagesize[dim]/2)+1);
				DATA_T phase = 2*M_PI*arg;
		
				// Fourier integration
				gridded_kdata.image[igrid] += (DATA_T)sampled_data.sqrtdcf[isample] * sampled_data.kdata[isample] * CPLX_T(cos(phase), sin(phase));
			}
		} */
		
	}
};
