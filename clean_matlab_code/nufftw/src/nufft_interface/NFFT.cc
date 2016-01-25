/* Michal Zarrouk, June 2013 */

#include "nufft_gen.h"
#include "nufft_util.h"

#include "nfft3util.h"
#include "nfft3.h"
#include "infft.h"

int main(int argc, char* argv[])
{
	/* usage:
		NFFT Nthreads direction imagefilename kdatafilename Ndim imagesize maxerr_power alpha trajfilename
	*/

	nuFFT_print("** NFFT **\n");
	
	
	// get input data
	int Nthreads = atoi(argv[1]);
	omp_set_num_threads(Nthreads);
	
	nuDFT_direction_t NFFT_direction = get_nuDFT_direction(argv[2]);
	
	char* imagefilename = argv[3];
	char* kdatafilename = argv[4];
	
	int Ndim = atoi(argv[5]);

	int* imagesize = (int*) calloc (Ndim, sizeof(int));

	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[6+dim]);
		
	DATA_T maxaliasingerror = pow(10.0,atof(argv[6+Ndim]));

	DATA_T alpha = atof(argv[7+Ndim]);

	char* trajfilename = argv[8+Ndim];
    
	
	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	

	
	double t0 = timestamp();
	nuFFT_print("transform initialization and data loading: ");
	
	INIT_FFTW;
	
	// load/create transform implementation data
	nfft_plan p;
	
	int* G = (int*) calloc (Ndim, sizeof(int));
	int Ntotal = 1;
	for (int dim=0; dim<Ndim; dim++) {
		G[dim] = ceil(alpha*imagesize[dim]);
		Ntotal *= imagesize[dim];
	}
	DATA_T beta;
	DATA_T width = KernelWidth (maxaliasingerror, alpha, imagesize[0], &beta)/2;
	
	int Nsamples = get_traj_nsamples(trajfilename, Ndim);
	
	nfft_init_guru(&p, Ndim, imagesize, Nsamples, G, width, PRE_PHI_HUT| PRE_FULL_PSI | MALLOC_X | MALLOC_F_HAT| MALLOC_F| FFT_OUT_OF_PLACE| FFTW_INIT, fftw_flags);
	free(imagesize);
	
	TRAJ_T* dcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
	read_trajectory_file_noalloc(trajfilename, Ndim, p.x, dcf, Nsamples);
	
	nfft_precompute_one_psi(&p);

	// load source data
	if (NFFT_direction == ADJOINT) {
		CPLX_T* noncartesian_data_nodcf = read_data_file(kdatafilename); // possible: use Ndata=Nsamples
		density_compensate_noalloc (noncartesian_data_nodcf, dcf, Nsamples, p.f);
		free(noncartesian_data_nodcf);
	}
	else if (NFFT_direction == FORWARD) {
		FILE* datafile = nufft_fopen(imagefilename,"rb");
		for (int igrid_rowmajor=0; igrid_rowmajor<Ntotal; igrid_rowmajor++)
		{
			int igrid_colmajor = reverse_storage_order (Ndim, imagesize, igrid_rowmajor);
			fread(p.f_hat[igrid_colmajor], sizeof(DATA_T), 2, datafile);
		}
		fclose(datafile);
	}	
	
	double t1 = timestamp();
	nuFFT_print_time(t1-t0);
	
	
	
	// transform data from source to target (using transform implementation)
	nuFFT_print("data transformation: ");
	if (NFFT_direction == ADJOINT)
		nfft_adjoint(&p);
	else if (NFFT_direction == FORWARD)
		nfft_trafo(&p);
	nuFFT_print_time(timestamp()-t1);
	
	
	// write transformed (target) data to file
	if (NFFT_direction == ADJOINT) {
		FILE* datafile = nufft_fopen(imagefilename,"wb");
		for (int igrid_rowmajor=0; igrid_rowmajor<Ntotal; igrid_rowmajor++) {
			int igrid_colmajor = reverse_storage_order (Ndim, imagesize, igrid_rowmajor);
			fwrite(p.f_hat[igrid_colmajor], 2, sizeof(DATA_T), datafile);
		}
		fclose(datafile);
	}
	else if (NFFT_direction == FORWARD)
		write_data_file(kdatafilename, p.f, Nsamples);
		
	nfft_finalize(&p);
	fftw_cleanup_threads();
	free(dcf);
	free(G);
	
	return 0;
}

