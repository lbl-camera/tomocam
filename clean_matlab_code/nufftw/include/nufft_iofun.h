/* Michal Zarrouk, June 2013 */

/* This file contains input and output functions for the nufft toolbox, for reading and writing from/to files. */

#include <cstdio>
#include <cstring>


FILE* nufft_fopen(char* filename, const char* mode)
{
	FILE* fid = fopen(filename, mode);
	if (fid == NULL)
		cerr << "Error: File " << filename << " can't be opened! Check whether the file exists.\n";	
	return fid;
};


int get_traj_nsamples(char* trajfilename, int Ndim)
{
	FILE* trajfile = nufft_fopen(trajfilename,"rb");
		
	// find number of sampled data points
	fseek(trajfile, 0, SEEK_END);
	int Nsamples = ftell(trajfile)/(Ndim+1)/sizeof(TRAJ_T);
	fclose(trajfile);
	
	return Nsamples;
};


int read_trajectory_file_noalloc (char* trajfilename, int Ndim, TRAJ_T*& trajectory, TRAJ_T*& sqrtdcf, int Nsamples)
{
	// read coordinates and dcfs from file
	FILE* trajfile = nufft_fopen(trajfilename,"rb");
	for (int isample=0; isample<Nsamples; isample++) {
		fread(&trajectory[isample*Ndim], sizeof(TRAJ_T), Ndim, trajfile);
		TRAJ_T a;
		fread(&a, sizeof(TRAJ_T), 1, trajfile);
		sqrtdcf[isample] = a;
	}
	fclose(trajfile);

	return 0;
};


/* read_trajectory_file - reads a stream of sample coordinates and the associated density compensation factors.

	Arguments:
	trajfilename - name of trajectory file
	Ndim - number of dimensions of the trajectory
	trajectory - (output) nonuniform sample coordinates
	sqrtdcf - (output) square root of density compensation factors
	Nsamples - (output) number of samples in trajectory
*/
int read_trajectory_file (char* trajfilename, int Ndim, TRAJ_T*& trajectory, TRAJ_T*& sqrtdcf, int& Nsamples)
{
	Nsamples = get_traj_nsamples(trajfilename, Ndim);
	
	trajectory = (TRAJ_T*) calloc (Nsamples*Ndim, sizeof(TRAJ_T));
	sqrtdcf    = (TRAJ_T*) calloc (Nsamples,      sizeof(TRAJ_T));
	
	read_trajectory_file_noalloc(trajfilename, Ndim, trajectory, sqrtdcf, Nsamples);

	return 0;
};


int get_datafile_ndata(char* datafilename)
{
	FILE* datafile = nufft_fopen(datafilename,"rb");
	
	fseek(datafile, 0, SEEK_END);
	int Ndata = ftell(datafile)/sizeof(CPLX_T);
	fclose(datafile);
	
	return Ndata;
}


int read_data_file_noalloc (char* datafilename, CPLX_T*& data, int Ndata)
{
	FILE* datafile = nufft_fopen(datafilename,"rb");
	fread(data, sizeof(DATA_T), Ndata*2, datafile);
	fclose(datafile);
	
	return 0;
}


/* read_data_file - reads a stream of pairs of real and imaginary parts of data

	Arguments:
	datafilename - name of data file
	Ndata - number of data points in data
	data - (output) complex data read from the file
*/
CPLX_T* read_data_file (char* datafilename, int Ndata=-1)
{
	if (Ndata < 0) // find number of sampled data points
		Ndata = get_datafile_ndata(datafilename);
	
	CPLX_T* data = (CPLX_T*) calloc (Ndata, sizeof(CPLX_T));
	
	read_data_file_noalloc(datafilename, data, Ndata);
	
	return data;
};





/* write_data_file - writes a vector of complex data to file

	Arguments:
	datafilename - name of destination file
	data - complex data vector
	Ndata - number of samples
*/
int write_data_file (char* datafilename, CPLX_T* data, int Ndata)
{
	FILE* datafile = nufft_fopen(datafilename,"wb");
	fwrite((DATA_T*)data, Ndata*2, sizeof(DATA_T), datafile);
	fclose(datafile);
	
	return 0;
};


// this is similar to the ind2sub function from MATLAB
int* num_to_matrix_indices (int Ndim, int* matrixsize, int num)
{
	int* indices = (int*) calloc (Ndim, sizeof(int));
	int dim = 0;
	for (int q=num; dim<Ndim; q = q/matrixsize[dim], dim++)
		indices[dim] = q % matrixsize[dim];
	return indices;
}


/* nuDFT_direction_t - direction of the non uniform Fourier transform.

	FORWARD - forward nuDFT, from the uniformly sampled (image) space to the non-uniformly sampled Fourier (k-) space.
	ADJOINT - adjoint nuDFT, from the non-uniformly sampled Fourier (k-) space to the uniformly sampled (image) space.
*/
enum nuDFT_direction_t {FORWARD, ADJOINT};

nuDFT_direction_t get_nuDFT_direction(char* nuDFT_direction)
{
	if (strcmp(nuDFT_direction,"forward") == 0)
		return FORWARD;
	
	else if (strcmp(nuDFT_direction,"adjoint") == 0)
		return ADJOINT;
}




/* density_compensate - point-wise multiplies an array of complex numbers by an array of real numbers,
	effectively density-compensating the data

	Arguments:
	data - complex data vector
	Nsamples - number of samples in data
	sqrtdcf - vector of square root of density compensation factors
*/
int density_compensate_noalloc (CPLX_T* data_nodcf, DATA_T* sqrtdcf, int Nsamples, CPLX_T*& density_compensated_data)
{
	for (int isample=0; isample<Nsamples; isample++) {
		density_compensated_data[isample][0] = data_nodcf[isample][0] * sqrtdcf[isample];
		density_compensated_data[isample][1] = data_nodcf[isample][1] * sqrtdcf[isample];
	}
	return 0;
}

CPLX_T* density_compensate (CPLX_T* data_nodcf, DATA_T* sqrtdcf, int Nsamples)
{
	CPLX_T* density_compensated_data = (CPLX_T*) calloc (Nsamples, sizeof(CPLX_T));
	density_compensate_noalloc (data_nodcf, sqrtdcf, Nsamples, density_compensated_data);
	return density_compensated_data;
}

int reverse_storage_order (int Ndim, int* matrixsize, int source_index)
{
	int dim = 0;
	int target_index = 0;
	for (int q=source_index; dim<Ndim; q = q/matrixsize[dim], dim++)
		target_index = q % matrixsize[dim] + matrixsize[dim]*target_index;
	return target_index;
}

