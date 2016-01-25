/* Michal Zarrouk, June 2013 */

#include "nufft.h"

int main(int argc, char* argv[])
{
	// usage: create_nufft_impfile Nthreads Ndim imagesize maxaliasingerror_power alpha resampling_method trajfilename impfilename

	nuFFT_print("** create nuFFT implementation file **\n");

	// get input data
	int Nthreads = atoi(argv[1]);
	omp_set_num_threads(Nthreads);
	
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
    
    DATA_T maxaliasingerror = pow(10.0,atof(argv[3+Ndim]));
	
	DATA_T alpha = atof(argv[4+Ndim]);

	resampling_method_t resampling_method = get_resampling_method(argv[5+Ndim]);
	
	char* trajfilename = argv[6+Ndim];
	
	char* impfilename = argv[7+Ndim];
		
	
	nuFFT_implementation_t nuFFT_imp(Ndim, imagesize, maxaliasingerror, alpha, resampling_method, trajfilename);

	nuFFT_imp.compute();
	
	nuFFT_imp.write_impfile(impfilename);
	
	nuFFT_imp.Free();
	free(imagesize);
	
	return 0;
}



