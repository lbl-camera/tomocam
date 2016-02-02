/* Michal Zarrouk, June 2013 */

#include "nudft.h"



int main(int argc, char* argv[])
{
	/* usage:
		nudft Nthreads direction imagefilename kdatafilename Ndim imagesize trajfilename
	*/


	nuFFT_print("** nuDFT **\n");
	
	int Nthreads = atoi(argv[1]);
	omp_set_num_threads(Nthreads);
	
	nuDFT_direction_t nuDFT_direction = get_nuDFT_direction(argv[2]);
	
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
	nuDFT_implementation_t nuDFT_imp(trajfilename, Ndim, imagesize);
	free(imagesize);
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	
	if (nuDFT_direction == ADJOINT) {
		CPLX_T* noncartesian_data_nodcf = read_data_file(kdatafilename); // possible: use Ndata=nuDFT_imp.Nsamples
		noncartesian_data = density_compensate(noncartesian_data_nodcf, nuDFT_imp.sqrtdcf, nuDFT_imp.Nsamples);
		free(noncartesian_data_nodcf);
		
		cartesian_data = (CPLX_T*) calloc (nuDFT_imp.Ntotal, sizeof(CPLX_T));
	}
	else if (nuDFT_direction == FORWARD) {
		cartesian_data = read_data_file(imagefilename); // possible: use Ndata=direct_nuDFT_imp.Ntotal
		noncartesian_data = (CPLX_T*) calloc (nuDFT_imp.Nsamples, sizeof(CPLX_T));
	}
	
	double t1 = timestamp();
	nuFFT_print_time(t1-t0);
	
	// transform data from source to target (using transform implementation)
	if (nuDFT_direction == ADJOINT)
		nuDFT_imp.adjoint(noncartesian_data, cartesian_data);
	else if (nuDFT_direction == FORWARD)
		nuDFT_imp.forward(cartesian_data, noncartesian_data);
	
	
	// write transformed (target) data to file
	if (nuDFT_direction == ADJOINT)
		write_data_file(imagefilename, cartesian_data, nuDFT_imp.Ntotal);
	else if (nuDFT_direction == FORWARD)
		write_data_file(kdatafilename, noncartesian_data, nuDFT_imp.Nsamples);
	
	free(noncartesian_data);
	free(cartesian_data);
	
	return 0;
}
