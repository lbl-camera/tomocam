/* Michal Zarrouk, June 2013 */

/* This file contains general, mathematical functions used in the nufft,
for example Kaiser-Bessel kernel calculations, etc. */

/* Many of the functions are based on the following paper:
	"Rapid Gridding Reconstruction With a Minimal Oversampling Ratio",
	P.J. Beatty, D.G. Nishimura and J.M. Pauly,
	IEEE Transactions on Medical Imaging, Vol. 24, No. 6, June 2005.
*/

#include <limits>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))


/* kernel_interpolation_method_t - specifying how the presampled kernel is interpolated onto the grid.
	
	NEAREST_NEIGHBOR - interpolated value is the value of the nearest presampled kernel sample.
	LINEAR           - value is linear interpolation between the two closest presampled kernel samples.
*/
enum kernel_interpolation_method_t {NEAREST_NEIGHBOR, LINEAR};


template<typename real_t>
real_t KernelWidth (real_t gridaccu, real_t alpha, int imagesize, real_t* beta, real_t Wmin=1, real_t Wmax=10, real_t gridaccutol=1E-3);

template<typename real_t>
real_t MaxAliasingAmplitude (real_t alpha, real_t W, int imagesize, real_t* beta);

template<typename real_t>
int OversampledImageSize(real_t alpha, int imagesize);

template<typename real_t>
real_t AliasingAmplitude (real_t pixel, real_t W, int G, real_t beta);

template<typename real_t>
real_t KaiserBesselOptimalBeta (real_t W, real_t alpha);

template<typename real_t>
real_t KaiserBessel_ImageDomain (real_t x, real_t W, int G, real_t beta);

template<typename real_t>
real_t KaiserBessel_FourierDomain (real_t k, real_t W, int G, real_t beta);

template<typename real_t>
real_t KernelSamplingRatio (real_t maxaliasingamp, real_t alpha, kernel_interpolation_method_t kernel_interpolation_method);

double BESSI0(double X);


/* KernelWidth - calculate the gridding interpolation kernel width that corresponds to a given grid
	oversampling ratio and maximum aliasing amplitude, as defined by Beatty et al. in formula (3)
	(see fig. 3).
	
	Arguments:
	maa - maximum aliasing amplitude of gridding
	alpha - grid oversampling ratio
	imagesize - number of pixels in image
	beta - (output) optimal Kaiser-Bessel kernel shape factor that corresponds to alpha and W
	Wmin, Wmax - define the search range for the kernel width
	maatol - percentage of error tolerance allowed for deviation from maa	
*/
template<typename real_t>
real_t KernelWidth (real_t maa, real_t alpha, int imagesize, real_t* beta, real_t Wmin=1, real_t Wmax=10, real_t maatol=1E-3)
{
	real_t W; // width of gridding interpolation kernel, in pixel units
	real_t curmaa = (real_t)std::numeric_limits<real_t>::max(); // maximum aliasing amplitude of current estimate of kernel width

	// Binary search for the kernel width that corresponds to alpha and maa
	while ( fabs((curmaa - maa)/maa) > maatol ) {
		W = (Wmax + Wmin)/2;
		curmaa = MaxAliasingAmplitude(alpha, W, imagesize, &*beta);
		
		if (curmaa > maa)
			Wmin = W;
		else
			Wmax = W;
		// ((curmaa > maa) ? Wmin : Wmax) = W;
	}
	
	return W;
}


/* MaxAliasingAmplitude - maximum aliasing amplitude of gridding with a given grid oversampling ratio and kernel width,
	as defined by Beatty et al. in formula (3).
	
	Arguments:
	alpha - grid oversampling ratio
	W - kernel width in pixel units
	imagesize - number of pixels in image
	beta - (output) optimal Kaiser-Bessel kernel shape factor for gridding with alpha and W
*/
template<typename real_t>
real_t MaxAliasingAmplitude (real_t alpha, real_t W, int imagesize, real_t* beta)
{
	int G = OversampledImageSize(alpha, imagesize); // number of pixels in oversampled image
	*beta = KaiserBesselOptimalBeta(W,alpha); // optimal Kaiser-Bessel kernel shape factor for gridding with alpha and W
	real_t maxaliasingamp = 0; // maximum aliasing amplitude
	real_t curamp; // current maximum aliasing amplitude

	// find maximum aliasing amplitude over image pixels
	for (int x = -floor(imagesize/2); x < ceil(imagesize/2); x++) {
		curamp = AliasingAmplitude ((real_t)x, W, G, *beta);
		maxaliasingamp = MAX(maxaliasingamp,curamp);
	}
	return maxaliasingamp;
}


/* OversampledImageSize - size of oversampled image in pixel units.
	We enforce an even number so that the phase modulation (which is necessary to avoid
	fftshifting after the fft) will be just +-1, and not complex (which would require a
	double amount of storage for the sparse convolution matrix, and complex multiplication).
*/
template<typename real_t>
int OversampledImageSize (real_t alpha, int imagesize)
{
	int G = ceil(alpha*imagesize);
	if (G%2 == 1)
		G++;
	return G;
}


/* AliasingAmplitude - measure of a pixel's gridding accuracy, as defined by Beatty et al. in formula (3).

	Arguments:
	x - pixel number within image
	W - kernel width
	G - oversampled image size
	beta - Kaiser-Bessel kernel shape factor
*/
template<typename real_t>
real_t AliasingAmplitude (real_t x, real_t W, int G, real_t beta)
{
	/* number of aliasing replications to sum (see formula (3) in Beatty et al.). This parameter is actually closely related to maatol from
	the KernelWidth function. In order to get maa = 1e-3, we need around 6 replications from each side.*/
	int sumwidth = 6;
	
	real_t argsum = 0; // numerator of formula (3)
	for (int p=1; p<=sumwidth; p++)
		argsum += pow(KaiserBessel_ImageDomain(x+G*p, W, G, beta),2)
		       +  pow(KaiserBessel_ImageDomain(x-G*p, W, G, beta),2);
		       
	real_t a = KaiserBessel_ImageDomain(x, W, G, beta); // denominator of formula (3)
	
	return sqrt(argsum)/a; // aliasing amplitude
}


/* KaiserBesselOptimalBeta - Optimal shape factor for gridding with the Kaiser-Bessel kernel,
	as defined by Beatty et al. in formula (5).
	
	Arguments:
	W - width of the gridding interpolation kernel
	alpha - grid oversampling ratio
*/
template<typename real_t>
real_t KaiserBesselOptimalBeta (real_t W, real_t alpha)
{
	return M_PI*sqrt(pow(W/alpha*(alpha-.5),2)-0.8);
}


/* KaiserBessel_ImageDomain - Kaiser-Bessel function in the image domain,
	as presented by Beatty et al. in formula (4).
	
	Arguments:
	x - pixel number within image
	W - Kaiser-Bessel window width in pixel units
	G - oversampled image size
	beta - Kaiser-Bessel window shape factor
*/
template<typename real_t>
real_t KaiserBessel_ImageDomain (real_t x, real_t W, int G, real_t beta)
{
	real_t arg = pow(M_PI*W*x/G,2)-pow(beta,2);
	if (arg >= 0) {
		real_t arg2 = sqrt(arg);
		return sin(arg2)/arg2;
	}
	else { // this is done in order to avoid complex numbers in the sine. The output will be real anyway.
		real_t arg2 = sqrt(-arg);
		return sinh(arg2)/arg2;
	}
}


/* KaiserBessel_FourierDomain - Kaiser-Bessel function in the Fourier domain,
	as presented by Beatty et al. in formula (4).
	
	Arguments:
	k - spatial frequency (units 1/pixel)
	W - Kaiser-Bessel window width in pixel units
	G - oversampled image size
	beta - Kaiser-Bessel window shape factor
*/
template<typename real_t>
real_t KaiserBessel_FourierDomain (real_t k, real_t W, int G, real_t beta)
{
	real_t arg = 2*k*G/W;
	if (arg < -1)
		return 0;
	real_t arg2 = 1-arg*arg;
	if (arg2 == 0) // it's not really zero, annoying numerical thing
		arg2 += 1E-10;
	return G/W*BESSI0(beta*pow(arg2,0.5));
}


/* KernelSamplingRatio - Optimal sampling ratio for the gridding interpolation kernel
 	as presented by Beatty et al. in formulas (7) and (8).
	
	Arguments:
	maxaliasingamp - maximum aliasing amplitude of gridding, as defined in Beatty et al., formula (3).
	alpha - grid oversampling ratio
	kernel_interpolation_method - interpolation method of the presampled gridding interpolation kernel
*/
template<typename real_t>
real_t KernelSamplingRatio (real_t maxaliasingamp, real_t alpha, kernel_interpolation_method_t kernel_interpolation_method)
{
	if (kernel_interpolation_method == NEAREST_NEIGHBOR)
		return 0.91/maxaliasingamp/alpha;
	
	if (kernel_interpolation_method == LINEAR)
		return sqrt(0.37/maxaliasingamp)/alpha;
}



/*
int* get_cummatrixsize(int Ndim, int* matrixsize)
{
	int* cummatrixsize = (int*) calloc (Ndim+1, sizeof(int));
	cummatrixsize[0] = 1;
	for (int dim=1; dim<=Ndim; dim++)
		cummatrixsize[dim] = cummatrixsize[dim-1]*matrixsize[dim-1];
	return cummatrixsize;
}

int* get_reverse_cummatrixsize(int Ndim, int* matrixsize)
{
	int* reverse_cummatrixsize = (int*) calloc (Ndim+1, sizeof(int));
	reverse_cummatrixsize[0] = 1;
	for (int dim=1; dim<=Ndim; dim++)
		reverse_cummatrixsize[dim] = reverse_cummatrixsize[dim-1]*matrixsize[Ndim-dim];
	return reverse_cummatrixsize;
}


int matrix_indices_to_num (int Ndim, int* matrixsize, int* indices)
{
	int num = 0;
	for (int dim=Ndim-1; dim>=0; dim--)
		num = indices[dim]+matrixsize[dim]*num;
	return num;
}

int* num_to_reverse_matrix_indices (int Ndim, int* matrixsize, int num)
{
	int* indices = (int*) calloc (Ndim, sizeof(int));
	int dim = 0;
	for (int q=num; dim<Ndim; q = q/matrixsize[dim], dim++)
		indices[Ndim-dim-1] = q % matrixsize[dim];
	return indices;
}
*/

/*
// I like to do things the really hard way.
int* num_to_matrix_indices (int Ndim, int* cummatrixsize, int num)
{
	int* indices = (int*) calloc (Ndim, sizeof(int));
	int dim = Ndim-1;
	for (int q=num; dim>=0; dim--, q %= cummatrixsize[dim])
		indices[dim] = q/cummatrixsize[dim];
	return indices;
}
int* num_to_reverse_matrix_indices(int Ndim, int* reverse_cummatrixsize, int num)
{
	int* reverse_indices = (int*) calloc (Ndim, sizeof(int));
	int dim = Ndim-1;
	for (int q=num; dim>=0; dim--, q %= reverse_cummatrixsize[dim])
		reverse_indices[Ndim-dim-1] = q/reverse_cummatrixsize[dim];
	return reverse_indices;
}
int reverse_endian_num (int Ndim, int* matrixsize, int* reverse_cummatrixsize, int source_endian_num)
{
	int* target_endian_indices = num_to_reverse_matrix_indices (Ndim, reverse_cummatrixsize, source_endian_num);
	int target_endian_num = matrix_indices_to_num (Ndim, matrixsize, target_endian_indices);
	free(target_endian_indices);
	return target_endian_num;
}
int reverse_endian_num (int Ndim, int* matrixsize, int source_endian_num)
{
	int* target_endian_indices = num_to_reverse_matrix_indices (Ndim, matrixsize, source_endian_num);
	int target_endian_num = matrix_indices_to_num (Ndim, matrixsize, target_endian_indices);
	free(target_endian_indices);
	return target_endian_num;
}
*/

