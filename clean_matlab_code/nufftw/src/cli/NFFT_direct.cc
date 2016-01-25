/* Michal Zarrouk, June 2013 */

#include <pthread.h>
pthread_attr_t gomp_thread_attr;

#include "nufft_gen.h"
#include "nufft_util.h"

#include "nfft3util.h"
#include "nfft3.h"
#include "infft.h"

int main(int argc, char* argv[])
{
	// usage: NFFT_direct Nthreads direction imagefilename kdatafilename Ndim imagesize trajfilename

	nuFFT_print("** direct NFFT **\n");
	
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
	
	char* trajfilename = argv[6+Ndim];
    

	
	double t0 = timestamp();
	nuFFT_print("transform initialization and data loading: ");
	
	// load/create transform implementation data
	nfft_plan p;
	
	int Ntotal = 1;
	for (int dim=0; dim<Ndim; dim++)
		Ntotal *= imagesize[dim];

	int Nsamples = get_traj_nsamples(trajfilename, Ndim);
	
	nfft_init(&p, Ndim, imagesize, Nsamples);
	
	TRAJ_T* dcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
	read_trajectory_file_noalloc(trajfilename, Ndim, p.x, dcf, Nsamples);
	
	// load source data & create target data
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
		nfft_adjoint_direct(&p);
	else if (NFFT_direction == FORWARD)
		nfft_trafo_direct(&p);
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
	free(imagesize);
	free(dcf);
	
	return 0;

}

