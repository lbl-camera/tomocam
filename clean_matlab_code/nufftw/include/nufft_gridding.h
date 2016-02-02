/* Michal Zarrouk, June 2013 */

/* This file implements the calculation of the gridding interpolation coefficients. */

/* presampled_kernel_t - presampled kernel class, contains the presampled kernel's values, its radius and the increment between samples. */
struct presampled_kernel_t
{
	/* members */
	TRAJ_T kerrad;    // kernel radius
	TRAJ_T kerinc;    // kernel increment
	DATA_T* kertable; // presampled kernel values
	
	
	/* methods */
	// constructor:
	presampled_kernel_t (int imagesize, DATA_T W, DATA_T beta, DATA_T alpha, DATA_T maxaliasingerror, int G)
	{ Init(imagesize, W, beta, alpha, maxaliasingerror, G); }
	
	// presampled_kernel_t::Init - initializes the presampled kernel with kernel information
	int Init (int imagesize, DATA_T W, DATA_T beta, DATA_T alpha, DATA_T maxaliasingerror, int G)
	{
		// k-space kernel radius, in k-space units (1/pixel)
		kerrad = W/2/G;
		
		// sampling ratio of the gridding interpolation kernel
		DATA_T S = KernelSamplingRatio(maxaliasingerror, alpha, LINEAR);

		// increment of presampled kernel values, in k-space units (1/pixel)
		kerinc = 1/S/G;
		
		// number of presampled kernel points
		int nker = (int)floor(S*W)+3;
		
		kertable = (DATA_T*) calloc (nker, sizeof(DATA_T));
		int iker = 0;
		for (DATA_T kker = -kerrad-kerinc; kker <= kerrad; kker += kerinc, iker++)
			kertable[iker] = KaiserBessel_FourierDomain (kker, W, G, beta);
		kertable[iker] = 0;
		
		/*DATA_T kker = -kerrad-kerinc;
		for (int iker=0; iker < nker-1; iker++, kker+= kerinc)
			kertable[iker] = KaiserBessel_FourierDomain (kker, W, G, beta);	
		kertable[iker] = 0;*/
		
		return 0;
	}
	
	// destructor:
	~presampled_kernel_t()
	{
		free(kertable);
	}
};



/* sampled_gridding_info_t - information about the gridding pattern of a nonuniform sample -
	the coordinates of the area it affects in the oversampled uniform grid, and the kernel interpolation/convolution
	values that it gives in that area.
	
	Interpolation is performed seperately in each dimension (such that the kernel is a multidimensional box,
	not radial, for example).
	This way, the kernel value on a grid point is a product of it's values in all dimensions, so we can
	compute a vector of grid kernel values for each dimension, and then just multiply the corresponding
	elements per each grid point.
	This is kind of like [x,y] = meshgrid(x,y) and then x.*y,
	but we do it like this in order to be able to perform for any number of dimensions.
*/
struct sample_gridding_info_t
{
	/* members */
	int* imin; // multidimensional index of first sample in oversampled uniform grid, which is within kernel radius w.r.t nonuniform sample
	int* nelemcum; // number of elements within kernel radius w.r.t nonuniform sample, accumulated along dimensions
	DATA_T** kervec; // kernel values on oversampled uniform grid locations w.r.t nonuniform sample, for each dimension
	int Ndim; // number of dimensions
	TRAJ_T sqrtdcf; // square root of sample's density compensation factor


	
	/* methods */
	// null constructor:
	sample_gridding_info_t(){}
	
	
	// constructor:
	sample_gridding_info_t (int ndim, TRAJ_T* kcoord, TRAJ_T sqrtdenscomp, presampled_kernel_t* presampled_kernel_alldims, int* G)
	{ Init(ndim, kcoord, sqrtdenscomp, presampled_kernel_alldims, G); }
	
	
	/* sample_gridding_info_t::Init - computes the affected grid area and associated kernel values in each dimension
	
		ndim - number of dimensions
		kcoord - coordinates of nonuniform sample
		sqrtdenscomp - square root of density compensation factor
		presampled_kernel_alldims - presampled interpolation kernel, for every dimension
		G - oversampled grid size in each dimension
	*/
	int Init (int ndim, TRAJ_T* kcoord, TRAJ_T sqrtdenscomp, presampled_kernel_t* presampled_kernel_alldims, int* G)
	{
		Ndim = ndim;
		sqrtdcf = sqrtdenscomp;
		
		/* compute affected grid area size and location */
			
		// location of first oversampled uniform grid point which is within kernel radius w.r.t sample in kcoord
		imin       = (int*) calloc (Ndim,   sizeof(int));
		
		// number of elements within kernel radius, in each dimension
		int* nelem = (int*) calloc (Ndim,   sizeof(int));
		
		// cumulative number of elements within kernel radius, along dimensions
		nelemcum   = (int*) calloc (Ndim+1, sizeof(int));
		
		nelemcum[0] = 1;
		for (int dim=0; dim<Ndim; dim++) { // for each dimension, calculate
			
			if (fabs(kcoord[dim])>0.5)
				nuFFT_print("k-space coordinates are beyond [-0.5 0.5]\n");
			
			// index of first sample in oversampled grid which is within kernel radius w.r.t sample in location kcoord, in this dimension
			imin[dim] = MAX(ceil ((kcoord[dim]-presampled_kernel_alldims[dim].kerrad+0.5)*G[dim]),0);
			
			// index of last sample in oversampled grid which is within kernel radius w.r.t sample in location kcoord, in this dimension
			int imax  = MIN(floor((kcoord[dim]+presampled_kernel_alldims[dim].kerrad+0.5)*G[dim]),G[dim]-1);
			
			// number of grid elements within kernel radius w.r.t sample in location kcoord, in this dimension
			nelem[dim] = imax-imin[dim]+1;
			
			// cumulative number of elements within kernel radius, for this dimension
			nelemcum[dim+1] = nelemcum[dim]*nelem[dim];
		}
		
		
		/* compute convolution results in each dimension */
		kervec = (DATA_T**) calloc (Ndim, sizeof(DATA_T*));
		for (int dim=0; dim<Ndim; dim++) { // for each dimension
		
			kervec[dim] = (DATA_T*) calloc (nelem[dim], sizeof(DATA_T));
			
			for (int s=0; s<nelem[dim]; s++) { // calculate kernel values on grid w.r.t sample, in this dimension
			
				int l = imin[dim]+s; // index of grid point within oversampled uniform grid, in this dimension
				DATA_T kker = -0.5+(DATA_T)l/G[dim]; // grid points' coordinate in k-space (within the range [-0.5 0.5])
		
				/* nearest neighbor interpolation */
				/* int p = round((kker-kcoord[dim]+imp.presampled_kernel_alldims[dim].kerrad)/imp.presampled_kernel_alldims[dim].kerinc);
				kervec[dim][s] = imp.presampled_kernel_alldims[dim].kertable[p]; */
				
				/* linear interpolation */
				DATA_T ik = (kker-kcoord[dim]+presampled_kernel_alldims[dim].kerrad+presampled_kernel_alldims[dim].kerinc)/presampled_kernel_alldims[dim].kerinc;
				// index of highest presampled kernel location that's smaller than the distance between the nonuniform sample and the current grid location
				int p = floor(ik);
				DATA_T ikf = ik - p;
				kervec[dim][s] = presampled_kernel_alldims[dim].kertable[p]*(1-ikf)+presampled_kernel_alldims[dim].kertable[p+1]*ikf;
			}

		}
		
		free(nelem);
		
		return 0;
	}

	/* sample_gridding_info_t::compute_grid_point - for a given grid point, compute the effect of this sample on it in all dimensions,
		meaning multiplying the corresponding convolution results from each dimesion.
		This is kind of like [x,y] = meshgrid(x,y) and then x.*y,
		but we do it like this in order to be able to perform for any number of dimensions.
		
		r - index of grid point within the grid area affected by this sample
		G - oversampled nonuniform grid size in each dimension
		igrid - index of this grid point within the oversampled uniform grid
		value - interpolation coefficient of the current sample on this grid point
	*/
	void compute_grid_point (int r, int* G, int& igrid, DATA_T& value)
	{
		// doing it backwards to avoid allocating and freeing grid indices memory
		igrid = 0;
		value = sqrtdcf;
		int q = r;
		int lsum = 0;
		for (int dim=Ndim-1; dim>=0; dim--) {
			int s = floor((DATA_T)q/nelemcum[dim]); // location of this grid point within the affected oversampled grid area, in this dimension
			value *= kervec[dim][s]; // accumulate kernel interpolation coefficients along dimensions
			int l = imin[dim]+s; // location of this grid point within the oversampled grid, in this dimension
			lsum += l; // sum of indices along dimensions, for +-1 modulation
			igrid = l+G[dim]*igrid;
			q = q % nelemcum[dim];
		}
	
		// we modulate +-1 inside the convolution and before FFT in order to avoid fftshifting, which would require annoying index calculations
		if (lsum % 2 == 1)
			value *= -1;
		
		
		/* // outdated versions!
		for (int dim=Ndim-1; dim>0; dim--)
		{
			int s = floor((DATA_T)q/nelemcum[dim-1]);
			gammatemp *= kervec[dim][s];
			int l = imin[dim]+s;
			lsum += l;
			igrid = l+G[dim]*igrid;
			q = q % nelemcum[dim-1];
		}
		gammatemp *= kervec[0][q];
		int l = imin[0]+q;
		lsum += l;
		igrid = l+G[0]*igrid;
	
		// more comprehensible version
		int igrid = 0;
		DATA_T gammatemp = 1;
		DATA_T q = r;
		for (int dim=Ndim-1; dim>=0; dim--)
		{
			int s;
			if (dim > 0)
				s = floor((float)q/nelemcum[isample][dim]);
			else
				s = q;
			gammatemp *= kervec[dim][s];
			int l = imin[isample][dim]+s;
			igrid = l+G[dim]*igrid;
			q = fmod(q,nelemcum[isample][dim]);
		}
		
		// different version
		int igrid = 0;
		DATA_T gammatemp = 1;
		int dim = Ndim-1;
		for (DATA_T q=r; dim>=0; q = fmod(q,nelemcum[isample][dim]))
		{
			int s;
			if (dim > 0)
				s = floor((float)q/nelemcum[isample][dim]);
			else
				s = q;
			gammatemp *= kervec[dim][s];
			int l = imin[isample][dim]+s;
			igrid = l+G[dim]*igrid;
			dim--;
		} */
		
	}
	
	/* // copy assignment operator:
	sample_gridding_info_t& operator= ( sample_gridding_info_t other )
	{
		swap(imin, other.imin);
		swap(nelemcum, other.nelemcum);
		for (int dim=0; dim<Ndim; dim++)
			swap(kervec[dim], other.kervec[dim]);
		swap(kervec, other.kervec);
    		//return *this;
    	} */
	
	// destructor:
	~sample_gridding_info_t ()
	{
		free(imin);
		free(nelemcum);
		for (int dim=0; dim<Ndim; dim++)
			free(kervec[dim]);
		free(kervec);
	}
};


/* resampling_method_t - specifys what method is used for convolution.

	ONTHEFLYCONV    - On-the-fly convolution (no precomputation).
	PRECOMPUTEDCONV - Precomputed matrix convolution (convolution by sparse matrix and vector multiplication, full precomputation).
*/
enum resampling_method_t {ONTHEFLYCONV=1, PRECOMPUTEDCONV=2};


resampling_method_t get_resampling_method(char* nuFFT_method)
{
	if (strcmp(nuFFT_method,"onthefly") == 0)
		return ONTHEFLYCONV;
	
	else if (strcmp(nuFFT_method,"spm") == 0)
		return PRECOMPUTEDCONV;
}

