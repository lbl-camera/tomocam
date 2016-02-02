/* Michal Zarrouk, June 2013 */

#include "nufft_gen.h"
#include "nufft_util.h"

#include "nfft3util.h"
#include "nfft3.h"
#include "infft.h"

//#include <cstring>

int main(int argc, char* argv[])
{
	// usage: forward_NFFT_direct trajfilename Ndim imagesize imagefilename kdatafilename Nthreads

	// get input data
	char* trajfilename = argv[1];
    
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
    
    char* imagefilename = argv[3+Ndim];
	
	char* kdatafilename = argv[4+Ndim];
	
	int Nthreads = atoi(argv[5+Ndim]);
	
	
	
	cerr << "**direct NFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";
	
	omp_set_num_threads(Nthreads);
	
	// load/create transform implementation data
	nfft_plan p;
	
	int Ntotal = 1;
	for (int dim=0; dim<Ndim; dim++)
		Ntotal *= imagesize[dim];

	int Nsamples = get_traj_nsamples(trajfilename, Ndim);
	
	nfft_init(&p, Ndim, imagesize, Nsamples);
	
	TRAJ_T* dcf = (TRAJ_T*) calloc (Nsamples, sizeof(TRAJ_T));
	read_trajectory_file_noalloc(trajfilename, Ndim, p.x, dcf, Nsamples);
	
	// load source data
	FILE* datafile = nufft_fopen(imagefilename,"rb");
	for (int ilittle=0; ilittle<Ntotal; ilittle++)
	{
		int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
		fread(p.f_hat[ibig], sizeof(DATA_T), 2, datafile);
	}
	fclose(datafile);
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	nfft_trafo_direct(&p);
	cerr << timestamp()-t1 << "s\n";
	
	
	// write transformed (target) data to file
	write_data_file(kdatafilename, p.f, Nsamples);
	
	nfft_finalize(&p);
	free(imagesize);
	free(dcf);
	return 0;
}

