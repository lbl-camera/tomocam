/* Michal Zarrouk, June 2013 */

#include "nudft.h"
//#include <cstring>

int main(int argc, char* argv[])
{
	// usage: forward_nudft trajfilename Ndim imagesize imagefilename kdatafilename Nthreads

	// get input data
	char* trajfilename = argv[1];
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
	
    char* imagefilename = argv[3+Ndim];
	
	char* kdatafilename = argv[4+Ndim];
	
	int Nthreads = atoi(argv[5+Ndim]);
	
	
	
	cerr << "**nuDFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";
	
	// set number of threads for OpenMP
	omp_set_num_threads(Nthreads);
	
	// load/create transform implementation data
	nuDFT_implementation_t nuDFT_imp(trajfilename, Ndim, imagesize);
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	read_data_file(imagefilename, cartesian_data); // possible: use Ndata=nuDFT_imp.Ntotal
	noncartesian_data = (CPLX_T*) calloc (nuDFT_imp.Nsamples, sizeof(CPLX_T));
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	nuDFT_imp.forward(cartesian_data, noncartesian_data);
	cerr << timestamp()-t1 << "s\n";
	
	
	// write transformed (target) data to file
	write_data_file(kdatafilename, noncartesian_data, nuDFT_imp.Nsamples);
	
	
	free(imagesize);
	free(noncartesian_data);
	free(cartesian_data);
	return 0;	
}

