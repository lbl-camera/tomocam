/* Michal Zarrouk, June 2013 */

#include "nufft.h"
#include <cstring>


int get_nuDFT_method_t(char* nuDFT_method);
nuDFT_direction_t get_nuDFT_direction_t(char* direction);
int direct_nuDFT(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads);
int nuFFT       (char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads,
	DATA_T alpha, unsigned FFTW_flags, DATA_T maxaliasingerror, resampling_method_t resampling_method);
int NFFT(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads,
	DATA_T alpha, unsigned FFTW_flags, DATA_T maxaliasingerror);
int NFFT_direct(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads);


int main(int argc, char* argv[])
{
	// usage: nufft_standalone trajfilename Ndim imagesize imagefilename kdatafilename nuDFT_direction Nthreads maxaliasingerror_power alpha 

	// get input data
	char* trajfilename = argv[1];
    
    int Ndim = atoi(argv[2]);
	
	int* imagesize = (int*) calloc (Ndim, sizeof(int));
	
	for (int dim=0; dim<Ndim; dim++)
		imagesize[dim] = atoi(argv[3+dim]);
    
    char* imagefilename = argv[3+Ndim];
	
	char* kdatafilename = argv[4+Ndim];
	
	nuDFT_direction_t nuDFT_direction = get_nuDFT_direction_t(argv[5+Ndim]);
	
	int Nthreads = atoi(argv[6+Ndim]);
	
	int nuDFT_method = get_nuDFT_method_t(argv[7+Ndim]);
	
	switch (nuDFT_method)
	{
		case 0:
			return direct_nuDFT(trajfilename, Ndim, imagesize, imagefilename, kdatafilename, nuDFT_direction, Nthreads);
			
		case ONTHEFLYCONV:
		case PRECOMPUTEDCONV:
		case 3: {
			DATA_T maxaliasingerror = pow(10.0,atof(argv[8+Ndim]));
	
			DATA_T alpha = atof(argv[9+Ndim]);

			unsigned FFTW_flags = FFTW_DESTROY_INPUT | FFTW_ESTIMATE; // estimate, measure, patient, exhaustive, wisdom_only
	
			switch (nuDFT_method)
			{
				case ONTHEFLYCONV:
				case PRECOMPUTEDCONV:
					return nuFFT(trajfilename, Ndim, imagesize, imagefilename, kdatafilename, nuDFT_direction, Nthreads, alpha, FFTW_flags, maxaliasingerror, (resampling_method_t)nuDFT_method);
			
				case 3:
					return NFFT (trajfilename, Ndim, imagesize, imagefilename, kdatafilename, nuDFT_direction, Nthreads, alpha, FFTW_flags, maxaliasingerror);
			}
		}
		
		case 4:
			return NFFT_direct (trajfilename, Ndim, imagesize, imagefilename, kdatafilename, nuDFT_direction, Nthreads);
	}
	
	return 0;
}


int get_nuDFT_method_t(char* nuDFT_method)
{
	if      (strcmp(nuDFT_method,"direct")      == 0)
		return 0;
	
	else if (strcmp(nuDFT_method,"onthefly")    == 0)
		return ONTHEFLYCONV;
	
	else if (strcmp(nuDFT_method,"spm")         == 0)
		return PRECOMPUTEDCONV;
	
	else if (strcmp(nuDFT_method,"nfft")        == 0)
		return 3;
	
	else if (strcmp(nuDFT_method,"nfft_direct") == 0)
		return 4;
}

nuDFT_direction_t get_nuDFT_direction_t(char* direction)
{
	if      (strcmp(direction,"forward") == 0)
		return FORWARD;
	
	else if (strcmp(direction,"adjoint") == 0)
		return ADJOINT;
}





int direct_nuDFT(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads)
{
	cerr << "**direct nuDFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization, and data loading: ";
	
	// set number of threads for OpenMP
	omp_set_num_threads(Nthreads);
	
	// load/create transform implementation data
	direct_nuDFT_implementation_t direct_nuDFT_imp(trajfilename, Ndim, imagesize);
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	if (nuDFT_direction == FORWARD)
	{
		read_data_file(imagefilename, cartesian_data); // possible: use Ndata=direct_nuDFT_imp.Ntotal
		noncartesian_data = (CPLX_T*) calloc (direct_nuDFT_imp.Nsamples, sizeof(CPLX_T));
	}
	else // nuDFT_direction == ADJOINT
	{
		read_data_file(kdatafilename, noncartesian_data); // possible: use Ndata=direct_nuDFT_imp.Nsamples
		cartesian_data = (CPLX_T*) calloc (direct_nuDFT_imp.Ntotal, sizeof(CPLX_T));
	}
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	if (nuDFT_direction == FORWARD)
		direct_nuDFT_imp.forward(cartesian_data, noncartesian_data);
	else // nuDFT_direction == ADJOINT
		direct_nuDFT_imp.adjoint(noncartesian_data, cartesian_data);
	cerr << timestamp()-t1 << "s\n";
	
	// write transformed (target) data to file
	if (nuDFT_direction == FORWARD)
		write_data_file(kdatafilename, noncartesian_data, direct_nuDFT_imp.Nsamples);
	else // nuDFT_direction == ADJOINT
		write_data_file(imagefilename, cartesian_data, direct_nuDFT_imp.Ntotal);
	
	return 0;
}


int nuFFT(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads,
	DATA_T alpha, unsigned FFTW_flags, DATA_T maxaliasingerror, resampling_method_t resampling_method)
{
	cerr << "**nuFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";

	// set number of threads for OpenMP	
	omp_set_num_threads(Nthreads);
	
	// load/create transform implementation data
	if (resampling_method == PRECOMPUTEDCONV)
		oski_Init();
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	nuFFT_implementation_t nuFFT_imp(trajfilename, Ndim, imagesize, maxaliasingerror, alpha, resampling_method, FFTW_flags, Nthreads);
	
	// load source data & create target data
	CPLX_T* cartesian_data;
	CPLX_T* noncartesian_data;
	if (nuDFT_direction == FORWARD)
	{
		read_data_file(imagefilename, cartesian_data); // possible: use Ndata=direct_nuDFT_imp.Ntotal
		noncartesian_data = (CPLX_T*) calloc (nuFFT_imp.Nsamples, sizeof(CPLX_T));
	}
	else // nuDFT_direction == ADJOINT
	{
		read_data_file(kdatafilename, noncartesian_data); // possible: use Ndata=direct_nuDFT_imp.Nsamples
		density_compensate(noncartesian_data, nuFFT_imp.Nsamples, nuFFT_imp.dcf);//adjust to case when file doesn't contain dcf and compute dcf ourselves.
		cartesian_data = (CPLX_T*) calloc (nuFFT_imp.Ntotal, sizeof(CPLX_T));
	}
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	if (nuDFT_direction == FORWARD)
		nuFFT_imp.forward(cartesian_data, noncartesian_data);
	else // nuDFT_direction == ADJOINT
		nuFFT_imp.adjoint(noncartesian_data, cartesian_data);
	cerr << timestamp()-t1 << "s\n";
	
	// write transformed (target) data to file
	if (nuDFT_direction == FORWARD)
		write_data_file(kdatafilename, noncartesian_data, nuFFT_imp.Nsamples);
	else // nuDFT_direction == ADJOINT
		write_data_file(imagefilename, cartesian_data, nuFFT_imp.Ntotal);
	
	return 0;
}



#include "nfft3util.h"
#include "nfft3.h"
#include "infft.h"

int NFFT(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads,
	DATA_T alpha, unsigned FFTW_flags, DATA_T maxaliasingerror)
{
	cerr << "**NFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";
	
	omp_set_num_threads(Nthreads);
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	
	// load/create transform implementation data
	nfft_plan p;
	
	int Nsamples;
	TRAJ_T* dcf;
	read_trajectory_file(trajfilename, Ndim, p.x, dcf, Nsamples);
	int* G = (int*) calloc (Ndim, sizeof(int));
	int Ntotal = 1;
	for (int dim=0; dim<Ndim; dim++)
	{
		G[dim] = ceil(alpha*imagesize[dim]);
		Ntotal *= imagesize[dim];
	}
	DATA_T beta;
	DATA_T width = KernelWidth (maxaliasingerror, alpha, imagesize[0], &beta)/2;

	nfft_init_guru(&p, Ndim, imagesize, Nsamples, G, width, PRE_PHI_HUT| PRE_FULL_PSI | MALLOC_F_HAT| MALLOC_F| FFT_OUT_OF_PLACE| FFTW_INIT, FFTW_flags);
	nfft_precompute_one_psi(&p);

	// load source data & create target data
	if (nuDFT_direction == FORWARD)
	{
		FILE* datafile = fopen(imagefilename,"rb");
		if (datafile == NULL)
			cerr << "Error: File " << imagefilename << " can't be openend! Check whether it exists.\n";
	
		for (int ilittle=0; ilittle<Ntotal; ilittle++)
		{
			int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
			fread(p.f_hat[ibig], sizeof(DATA_T), 2, datafile);
		}
		fclose(datafile);
	}
	else // nuDFT_direction == ADJOINT
	{
		read_data_file(kdatafilename, p.f);
		density_compensate(p.f, Nsamples, dcf);
	}
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	if (nuDFT_direction == FORWARD)
		nfft_trafo(&p);
	else // nuDFT_direction == ADJOINT
		nfft_adjoint(&p);
	cerr << timestamp()-t1 << "s\n";
	
	// write transformed (target) data to file
	if (nuDFT_direction == FORWARD)
		write_data_file(kdatafilename, p.f, Nsamples);
	else // nuDFT_direction == ADJOINT
	{
		FILE* datafile = fopen(imagefilename,"wb");
		if (datafile == NULL)
			cerr << "Error: File " << imagefilename << " can't be openend! Check whether the directory exists.\n";
		
		for (int ilittle=0; ilittle<Ntotal; ilittle++)
		{
			int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
			fwrite(p.f_hat[ibig], 2, sizeof(DATA_T), datafile);
		}
		fclose(datafile);
	}
	return 0;

}


int NFFT_direct(char* trajfilename, int Ndim, int* imagesize, char* imagefilename, char* kdatafilename, nuDFT_direction_t nuDFT_direction, int Nthreads)
{
	cerr << "**direct NFFT**\n";
	
	double t0 = timestamp();
	cerr << "transform initialization and data loading: ";
	
	omp_set_num_threads(Nthreads);
	//fftw_init_threads();
	//fftw_plan_with_nthreads(omp_get_max_threads());
	
	// load/create transform implementation data
	nfft_plan p;
	
	int Nsamples;
	TRAJ_T* dcf;
	read_trajectory_file(trajfilename, Ndim, p.x, dcf, Nsamples);
	int Ntotal = 1;
	for (int dim=0; dim<Ndim; dim++)
		Ntotal *= imagesize[dim];

	nfft_init(&p, Ndim, imagesize, Nsamples);
	read_trajectory_file(trajfilename, Ndim, p.x, dcf, Nsamples);
//	nfft_init_guru(&p, Ndim, imagesize, Nsamples, G, width, PRE_PHI_HUT| PRE_FULL_PSI | MALLOC_F_HAT| MALLOC_F| FFT_OUT_OF_PLACE| FFTW_INIT, FFTW_flags);
//	nfft_precompute_one_psi(&p);

	// load source data & create target data
	if (nuDFT_direction == FORWARD)
	{
		FILE* datafile = fopen(imagefilename,"rb");
		if (datafile == NULL)
			cerr << "Error: File " << imagefilename << " can't be opened! Check whether it exists.\n";
	
		for (int ilittle=0; ilittle<Ntotal; ilittle++)
		{
			int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
			fread(p.f_hat[ibig], sizeof(DATA_T), 2, datafile);
		}
		fclose(datafile);
	}
	else // nuDFT_direction == ADJOINT
	{
		read_data_file(kdatafilename, p.f);
		density_compensate(p.f, Nsamples, dcf);
	}
	
	double t1 = timestamp();
	cerr << t1-t0 << "s\n";
	
	
	// transform data from source to target (using transform implementation)
	cerr << "data transformation: ";
	if (nuDFT_direction == FORWARD)
		nfft_trafo_direct(&p);
	else // nuDFT_direction == ADJOINT
		nfft_adjoint_direct(&p);
	cerr << timestamp()-t1 << "s\n";
	
	// write transformed (target) data to file
	if (nuDFT_direction == FORWARD)
		write_data_file(kdatafilename, p.f, Nsamples);
	else // nuDFT_direction == ADJOINT
	{
		FILE* datafile = fopen(imagefilename,"wb");
		if (datafile == NULL)
			cerr << "Error: File " << imagefilename << " can't be openend! Check whether the directory exists.\n";
		
		for (int ilittle=0; ilittle<Ntotal; ilittle++)
		{
			int ibig = reverse_endian_num (Ndim, imagesize, ilittle);
			fwrite(p.f_hat[ibig], 2, sizeof(DATA_T), datafile);
		}
		fclose(datafile);
	}
	return 0;

}
