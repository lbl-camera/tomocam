/* Michal Zarrouk, July 2013 */

#include "nufft.h"

/* tuning_heuristic_t - specifys which tuning heuristic is used to decide on the optimal nuFFT implementation
	
	FFTTUNE   - Tunes the FFT(W) phase only - uses the implementation that's best for FFTW.
	NUFFTTUNE - Tunes the entire nuFFTW process: gridding (either on the fly or precomputed), FFTW and deapodization.
*/
enum tuning_heuristic_t {FFTWTUNE, NUFFTTUNE};

tuning_heuristic_t get_tuning_heuristic(char* tuning_heur)
{
	if (strcmp(tuning_heur,"fftw") == 0)
		return FFTWTUNE;
	
	else if (strcmp(tuning_heur,"nufft") == 0)
		return NUFFTTUNE;
}




#define Ntrials 10

/* nuFFT_tuner_t - Implementation search space data type, containing all nuFFT implementations to be evaluated during planning. */
struct nuFFT_tuner_t
{
	/* members */
	int Ndim;     // Number of dimensions of transform
	int* N;       // uniform grid size, an array of Ndim elements
	int Ntotal;   // total number of grid samples (Ntotal = prod(N))
	int Nsamples; // number of nonuniform samples
	TRAJ_T* trajectory; // nonuniform sample coordinates
	TRAJ_T* sqrtdcf;  // square root of density compensation factors for nonuniform samples
	
	bool free_trajectory;
	
	int Nimplementations; // number of implementations in search space
	nuFFT_implementation_t* implementations;
	int optimal_implementation_num;
	nuFFT_implementation_t optimal_implementation;
	
	
	/* methods */
	/*// default constructor:
	nuFFT_tuner_t()
	{
		int ndim = 2;
		int* imagesize = (int*) calloc (ndim, sizeof(int));
		for (int dim=0; dim<ndim; dim++)
			imagesize[dim] = 128;
		nuFFT_tuner_t(ndim, imagesize, 1E-3, 1.2, 2, 20, PRECOMPUTEDCONV, 12);
	}*/
	
	
	// constructor:
	nuFFT_tuner_t(int ndim, int* imagesize, DATA_T maxaliasingerror,
		DATA_T alphastart, DATA_T alphaend, int nalpha, resampling_method_t resampling_method, unsigned fftw_flags) :
			Ndim(ndim), Nimplementations(nalpha)
	{
		/*
			Implementations are defined by a set of oversampling ratio (alpha), maximum aliasing error (maxaliasingerror),
			and corresponding kernel width (W). Currently the implementation search space is defined by an input maxaliasingerror,
			and possible combinations of alpha and W are evaluated. The range of alphas to be evaluated will be
			linspace(alphastart, alphaend, Nalpha), so Nalpha also defines the number of implementations in the search space.
		*/

		
		nuFFT_init(resampling_method);
	
		N = (int*) calloc (Ndim, sizeof(int));
		Ntotal = 1;
		for (int dim=0; dim<Ndim; dim++) {
			N[dim] = imagesize[dim];
			Ntotal *= N[dim];
		}
		
		implementations = (nuFFT_implementation_t*) calloc (Nimplementations, sizeof(nuFFT_implementation_t));
		DATA_T alphaspacing = (alphaend-alphastart)/nalpha;

		#pragma omp parallel for
		for (int iimp = 0; iimp < Nimplementations; iimp++) {
			double alpha = alphastart + iimp*alphaspacing;
			implementations[iimp].set_params(Ndim, imagesize, maxaliasingerror, alpha, resampling_method);
		}
		
		
	}
	
	
	// destructor:
	~nuFFT_tuner_t()
	{
		implementations[optimal_implementation_num].Free();
		free(implementations);
		free(trajectory);
		free(sqrtdcf);
		free(N);
		nuFFT_close();
	}
	
	
	int tune(tuning_heuristic_t tuning_heuristic, char* trajfilename, unsigned fftw_flags, int npartitions)
	{
		read_trajectory_file(trajfilename, Ndim, trajectory, sqrtdcf, Nsamples);
		
		free_trajectory = true;
		
		return tune(tuning_heuristic, fftw_flags, npartitions);
	}
	
	int tune(tuning_heuristic_t tuning_heuristic, int nsamples, TRAJ_T* traj, TRAJ_T* denscomp, unsigned fftw_flags, int npartitions)
	{
		Nsamples = nsamples;
		trajectory = traj;
		sqrtdcf = denscomp;
		
		free_trajectory = false;
		
		return tune(tuning_heuristic, fftw_flags, npartitions);
	}
	
	int tune(tuning_heuristic_t tuning_heuristic, unsigned fftw_flags, int npartitions)
	{	
		// fabricate nonuniform data
		CPLX_T* noncartesian_data = (CPLX_T*) calloc(Nsamples, sizeof(CPLX_T));
		for (int isample=0; isample<Nsamples; isample++) {
			noncartesian_data[isample][0] = sqrtdcf[isample];
			noncartesian_data[isample][1] = sqrtdcf[isample];
		}
			
		// load source data & create target data
		CPLX_T* cartesian_data = (CPLX_T*) calloc (Ntotal, sizeof(CPLX_T));

		
		
		
		double optimal_t_nuFFT = (double)std::numeric_limits<double>::max();
		optimal_implementation_num = -1;
	
		for (int iimp=0; iimp<Nimplementations; iimp++)
		{
			nuFFT_print("implementation no. ");
			nuFFT_print(iimp);
			nuFFT_print("\n");
			
			double t_nuFFT_avg = 0;
			
			if (tuning_heuristic == FFTWTUNE) {
				implementations[iimp].tune_fftw(fftw_flags);
			
				for (int itrial=0; itrial < Ntrials; itrial++) {
					nuFFT_print("adjoint fftw: ");
					double t_nuFFT = implementations[iimp].fftw(ADJOINT);
					t_nuFFT_avg += t_nuFFT;
					nuFFT_print_time(t_nuFFT);
				}
				
				t_nuFFT_avg /= Ntrials;
				nuFFT_print("average runtime: ");
				nuFFT_print_time(t_nuFFT_avg);
			
				if (t_nuFFT_avg < optimal_t_nuFFT) {
					implementations[optimal_implementation_num].Free();
					optimal_t_nuFFT = t_nuFFT_avg;
					optimal_implementation_num = iimp;
				}
				else {
					implementations[iimp].Free();
				}
				
			}
			else if (tuning_heuristic = NUFFTTUNE) {
			
			
				implementations[iimp].trajectory = trajectory;
				implementations[iimp].sqrtdcf = sqrtdcf;
				implementations[iimp].Nsamples = Nsamples;
				implementations[iimp].compute();
				implementations[iimp].tune(fftw_flags, npartitions);
			
			
				for (int itrial=0; itrial < Ntrials; itrial++)
					t_nuFFT_avg += implementations[iimp].adjoint(noncartesian_data, cartesian_data);
			
				t_nuFFT_avg /= Ntrials;
				nuFFT_print("average runtime: ");
				nuFFT_print_time(t_nuFFT_avg);
			
				if (t_nuFFT_avg < optimal_t_nuFFT) {
					implementations[optimal_implementation_num].trajectory = NULL;
					implementations[optimal_implementation_num].sqrtdcf = NULL;
					implementations[optimal_implementation_num].Free();
					optimal_t_nuFFT = t_nuFFT_avg;
					optimal_implementation_num = iimp;
				}
				else {
					implementations[iimp].trajectory = NULL;
					implementations[iimp].sqrtdcf = NULL;
					implementations[iimp].Free();
				}
			}
		}
		
		optimal_implementation = implementations[optimal_implementation_num];
		
		nuFFT_print("optimal implementation is no. ");
		nuFFT_print(optimal_implementation_num);
		nuFFT_print("\n");
		
		free(cartesian_data);
		free(noncartesian_data);
		
		return optimal_implementation_num;
		
	}
	
	
};
