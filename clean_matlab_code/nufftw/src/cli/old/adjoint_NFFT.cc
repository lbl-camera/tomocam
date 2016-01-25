/* Michal Zarrouk, June 2013 */

#include "nufft_gen.h"
#include "nufft_util.h"

#include "nfft3util.h"
#include "nfft3.h"
#include "infft.h"

//#include <cstring>

int main(int argc, char* argv[])
{
	// usage: adjoint_NFFT trajfilename Ndim imagesize imagefilename kdatafilename Nthreads maxaliasingerror_power alpha

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

	unsigned fftw_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	
	
	cerr << "**NFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";
	
	omp_set_num_threads(Nthreads);
	
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	
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
	
	TRAJ_T* dcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
	read_trajectory_file_noalloc(trajfilename, Ndim, p.x, dcf, Nsamples);
	
	nfft_precompute_one_psi(&p);

	// load source data
	read_data_file(kdatafilename, p.f);
	density_compensate(p.f, Nsamples, dcf);
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	nfft_adjoint(&p);
	cerr << timestamp()-t1 << "s\n";
	
	
	// write transformed (target) data to file
	FILE* datafile = nufft_fopen(imagefilename,"wb");
	for (int ilittle=0; ilittle<Ntotal; ilittle++) {
		int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
		fwrite(p.f_hat[ibig], 2, sizeof(DATA_T), datafile);
	}
	fclose(datafile);
	
	nfft_finalize(&p);
	fftw_cleanup_threads();
	free(imagesize);
	free(dcf);
	free(G);
	return 0;
}

