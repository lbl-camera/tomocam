/* Michal Zarrouk, June 2013 */

/* This file handles parallel computation of the sparse gridding/convolution matrix. */

/* we need the <algorithm> library in order to sort the entries of the gridding/convolution sparse matrix,
so we can put it in CSR form after computing it in parallel. */
#include <algorithm>

extern "C" {
	#if DATA_TYPE == DOUBLE
		#include <oski/oski_Tid.h>
	#elif DATA_TYPE == FLOAT
		#include <oski/oski_Tis.h>
	#endif
}

/* merge - merge-sorts an array of sorted data sets
	
	Arguments:
	unsorted_vec - array of pointers to sorted sets of data
	nvec - number of data sets
	n - number of elements in each set
	ntotal - total number of elements in all sets
*/
template <typename T>
T* merge (T** unsorted_vec, int nvec, int n[], int ntotal)
{
	if (nvec == 1) {
		T* sorted = unsorted_vec[0];
		unsorted_vec[0] = NULL;
		return sorted;
	}
	
	// sorted data
	T* sorted = (T*) calloc (ntotal, sizeof(T));
	
	// offs indicates for each input array, the index of the first currently unsorted element
	int offs[nvec];
	for (int i = 0; i < nvec; ++i)
		offs[i] = 0;

	for (int n_sorted = 0; n_sorted < ntotal; ++n_sorted) { // for each element in output sorted array
	
		// find the next minimal element among all sets
		
		int p_min = -1; // index of set which contains the current minimal unsorted element
		T minval(0); // current minimal unsorted element

		for (int i = 0; i < nvec; ++i) // for each set
			if (offs[i] < n[i]) { // check whether this set still has unsorted elements
			
				T arg = unsorted_vec[i][offs[i]]; // current minimal unsorted element from this set
				
				if ( p_min == -1 || minval > arg ) { // if this element is lower than the current
					// make it the current minimal unsorted element
					minval = arg;
					p_min = i;
				}
			}
	
		offs[p_min]++; // update the current unsorted element of the set which contained the latest minimal element
		sorted[n_sorted] = minval;//.SpM_COO_entry_t(minval); // update the sorted array of the latest minimal element
	}
	
	return sorted;
}


/* sort_and_merge - merge-sorts an array of unsorted data sets
	
	Arguments:
	unsorted_vec - array of pointers to unsorted sets of data
	nvec - number of data sets
	n - number of elements in each set
	ntotal - total number of elements in all sets
*/
template <typename T>
T* sort_and_merge (T** unsorted_vec, int nvec, int n[], int ntotal)
{
	// sort each data set
	#pragma omp parallel for
	for (int ivec=0; ivec<nvec; ivec++)
		stable_sort(&unsorted_vec[ivec][0],&unsorted_vec[ivec][n[ivec]]);
	
	// merge-sort all data sets
	T* sorted = merge(unsorted_vec, nvec, n, ntotal);
	
	return sorted;
}


int* equalize(int M, int N)
{
	int mpern = M/N;
	int mmodn = M%N;
	int* mindices = (int*) calloc (N+1, sizeof(int));
	mindices[0] = 0;
	for (int iN=1; iN<N; iN++) {
		mindices[iN] = mindices[iN-1]+mpern;
		if (iN < mmodn)
			mindices[iN]++;
	}
	mindices[N] = M;
	return mindices;
}


/* SpM_COO_entry_t - sparse matrix entry in COO format (coordinate list) - row, column, and value,
	plus another member, ientry, for the location of an entry within a vector of entries, in order to sort the vector.
*/
struct SpM_COO_entry_t
{
	/* members */
	int row, col, ientry;
	DATA_T value;
	
	
	/* constructors */
	SpM_COO_entry_t (int x) : row(x), ientry(x) {}
	SpM_COO_entry_t (int r, int c, int i, DATA_T v) : row(r), col(c), ientry(i), value(v) {}
	SpM_COO_entry_t (SpM_COO_entry_t const& a) : row(a.row), col(a.col), ientry(a.ientry), value(a.value) {}
	
	/* operators */
	inline bool operator<(SpM_COO_entry_t const& b) const { return (row < b.row || (row == b.row && col < b.col)); }
	inline bool operator>(SpM_COO_entry_t const& b) const { return (row > b.row || (row == b.row && col > b.col)); }
};


/* SpM_COO_t - sparse matrix in COO (coordinate list) format. */
struct SpM_COO_t
{
	/* members */
	int Nrows; // number of rows in matrix
	int Ncols; // number of columns in matrix
	int nnz;   // number of entries in sparse matrix (number of nonzeros in matrix)
	SpM_COO_entry_t* entries; // sparse matrix elements
	bool sorted; // states whether the elements are sorted
	
	
	/* methods */
	// constructor:
	SpM_COO_t() : sorted(false) {};
	
	int sort ()
	{
		if (~sorted) {
			int Nthreads = omp_get_max_threads();
		
			int* threadentryind = equalize(nnz,Nthreads);
			int* n = (int*) calloc (Nthreads, sizeof(int));
			SpM_COO_entry_t** unsorted_entry_vec = (SpM_COO_entry_t**) calloc (Nthreads, sizeof(SpM_COO_entry_t*));
			for (int ithread=0; ithread<Nthreads; ithread++) {
				unsorted_entry_vec[ithread] = &entries[threadentryind[ithread]];
				n[ithread] = threadentryind[ithread+1]-threadentryind[ithread];
			}
			
			SpM_COO_entry_t* sorted_entries = sort_and_merge (unsorted_entry_vec, Nthreads, n, nnz);
			free(entries);
			entries = sorted_entries;
		
			sorted = true;
			
			free(threadentryind);
			free(n);
			free(unsorted_entry_vec);
		}
		return 0;	
	}
	
	
	// destructor:
	~SpM_COO_t()
	{
		free(entries);
	}
};


/* nuFFT_precompute_COO - computes entries of a sparse convolution matrix for non-uniform FFT
	
	Arguments:
	trajectory - nonuniform sample coordinates
	sqrtdcf - square root of density compensation factors
	Ndim - number of dimensions
	Nsamples - number of nonuniform samples
	presampled_kernel_alldims - presampled interpolation kernel, for every dimension
	G - oversampled grid size in each dimension
	Gtotal - total number of oversampled grid points (Gtotal=prod(G))
*/
SpM_COO_t nuFFT_precompute_COO (TRAJ_T* trajectory, TRAJ_T* sqrtdcf, int Ndim, int Nsamples, presampled_kernel_t* presampled_kernel_alldims, int* G, int Gtotal)
{
	int Nthreads = omp_get_max_threads();
	
	SpM_COO_entry_t** SpM_COO_entries_vec = (SpM_COO_entry_t**) calloc (Nthreads, sizeof(SpM_COO_entry_t*));
	
	// partition samples between threads
	int* threadsampleind = equalize(Nsamples,Nthreads);
	
	// estimate number of grid points within multi-dimensional kernel per sample
	int ngridpersample = 1;
	for (int dim=0; dim<Ndim; dim++)
		ngridpersample *= ceil(presampled_kernel_alldims[dim].kerrad*2*G[dim]);
	
	int* n = (int*) calloc (Nthreads, sizeof(int));
	#pragma omp parallel for
	for (int ithread=0; ithread<Nthreads; ithread++) {
		int nnzest = (threadsampleind[ithread+1]-threadsampleind[ithread]) * ngridpersample;
		SpM_COO_entries_vec[ithread] = (SpM_COO_entry_t*) calloc (nnzest, sizeof(SpM_COO_entry_t));
		int ientry = 0;
		for (int isample=threadsampleind[ithread]; isample<threadsampleind[ithread+1]; isample++) {
			sample_gridding_info_t sgi(Ndim, &trajectory[isample*Ndim], sqrtdcf[isample], presampled_kernel_alldims, G);
		
			for (int r=0; r<sgi.nelemcum[Ndim]; r++) { // for each grid point affected by this sample

				int igrid; // index of this grid point within the oversampled uniform grid
				DATA_T value; // interpolation coefficient of the current sample on this grid point
			
				// compute igrid and value
				sgi.compute_grid_point(r, G, igrid, value);
			
				// store this entry in the sparse matrix entry list
				SpM_COO_entries_vec[ithread][ientry] = SpM_COO_entry_t(igrid, isample, ientry, value);
				ientry++;
			}
		}
		n[ithread] = ientry;
	}
	
	SpM_COO_t SpM_COO;
	SpM_COO.nnz = 0;
	for (int ithread=0; ithread<Nthreads; ithread++)
		SpM_COO.nnz += n[ithread];
	
	// sort and merge
	SpM_COO.entries = sort_and_merge(SpM_COO_entries_vec, Nthreads, n, SpM_COO.nnz);
	
	SpM_COO.Nrows = Gtotal; // number of rows in the convolution matrix is the total number of oversampled grid points
	SpM_COO.Ncols = Nsamples; // number of columns in the convolution matrix is the number of nonuniform samples
	
	SpM_COO.sorted = true;
	
	
	free(n);
	for (int ithread=0; ithread<Nthreads; ithread++)
		free(SpM_COO_entries_vec[ithread]);
	free(SpM_COO_entries_vec);
	free(threadsampleind);
	return SpM_COO;
}


/* SpM_CSR_t - sparse matrix in compressed sparse row format (CSR). */
struct SpM_CSR_t
{
	/* members */
	int Nrows;
	int Ncols;
	int* row_offs;
	int* col_idcs;
	DATA_T* values;
	int nnz;
	int npartitions;
	oski_matrix_t Gamma;
	oski_vecview_t x;
	int **prow_offs;
	oski_matrix_t* pGamma;
	oski_vecview_t* py;
	int nch;
	//char* oski_wisdom;
	
	
	/* methods */
	// null constructor:
	SpM_CSR_t()
	{
		row_offs = NULL;
		col_idcs = NULL;
		values   = NULL;
		pGamma   = NULL;
		py       = NULL;
		Gamma    = NULL;
		x        = NULL;
		prow_offs = NULL;
		npartitions = 0;
		
	}
	
	// constructor:
	/*SpM_CSR_t(void* a)
	{
		row_offs = (int*)a;
		col_idcs = (int*)*(&a+1);
		values = (DATA_T*)*(&a+2);
		//oski_wisdom = (char*)*(&a+3);
		Gamma = (oski_matrix_t)*(a+3);
	}*/
	
	// constructor:
	SpM_CSR_t (SpM_COO_t& SpM_COO)
	{ COO2CSR(SpM_COO); }
	
	int COO2CSR (SpM_COO_t& SpM_COO)
	{
		nnz = SpM_COO.nnz;
		Nrows = SpM_COO.Nrows;
		Ncols = SpM_COO.Ncols;
		
		if (!SpM_COO.sorted)
			SpM_COO.sort();
		
		// calculate CSR
		row_offs = (int*) calloc (Nrows+1, sizeof(int));
		col_idcs = (int*) calloc (nnz, sizeof(int));
		values = (DATA_T*) calloc(nnz, sizeof(DATA_T));
	
		int irow = -1;
	
		for (int ientry = 0; ientry < nnz; ++ientry) {
			while (SpM_COO.entries[ientry].row > irow)
				row_offs[++irow] = ientry;
	
			col_idcs[ientry] = SpM_COO.entries[ientry].col;
			values[ientry]   = SpM_COO.entries[ientry].value;

			if (SpM_COO.entries[ientry].row != irow)
				row_offs[++irow] = ientry;
		}
	
		while (irow < Nrows)
			row_offs[++irow] = nnz;
	}
	
	int tune(int npart, CPLX_T* kgridded)
	{
		npartitions = npart;
		
		// partition matrix
		// try to equalize the number of nonzeros in each partition
		int p[npartitions+1];
		//int *ip[npartitions];
		int *jp[npartitions];
		DATA_T *vp[npartitions];
		
		
		int nnz_per_proc = ceil(nnz / npartitions);
		prow_offs = (int**) calloc(npartitions, sizeof(int*));
		int irow = 0;
		p[0] = 0;
		for (int ipartition = 0; ipartition < npartitions; ipartition++) {
			while (irow < Nrows && (row_offs[irow]-row_offs[p[ipartition]] <= nnz_per_proc))
				irow++;
			p[ipartition+1] = irow;
	
			int Nrows = p[ipartition+1] - p[ipartition];
			prow_offs[ipartition] = (int*) calloc(Nrows+1,sizeof(int));
			//ip[ipartition] = (int*) calloc(Nrows+1,sizeof(int));

			for (int irow = 0; irow <= Nrows; irow++)
				//ip[ipartition][irow] = row_offs[p[ipartition]+irow] - row_offs[p[ipartition]];
				prow_offs[ipartition][irow] = row_offs[p[ipartition]+irow] - row_offs[p[ipartition]];
		
			jp[ipartition] = &col_idcs[row_offs[p[ipartition]]];
			vp[ipartition] = &values[row_offs[p[ipartition]]];
		}
		
		nch = 1; /* this can be used in the future to simultaneously transform a number of data sets with the same implementation,
			by multiplying the sparse convolution matrix with a matrix of data vectors rather than just one.*/
		pGamma = (oski_matrix_t*) calloc (npartitions, sizeof(oski_matrix_t));
		py = (oski_vecview_t*) calloc (npartitions, sizeof(oski_vecview_t));
		
		Gamma = oski_CreateMatCSR (row_offs, col_idcs, values, Nrows, Ncols, SHARE_INPUTMAT,
		                       1, INDEX_ZERO_BASED);

		oski_SetHintMatMult (Gamma, OP_TRANS, 
			                 1.0, SYMBOLIC_MULTIVEC,
			                 0.0, SYMBOLIC_MULTIVEC, 100);

		//#pragma omp barrier
		oski_TuneMat (Gamma);
	
		x = oski_CreateMultiVecView ((DATA_T*)kgridded, Nrows, 2*nch, LAYOUT_ROWMAJ, 2*nch);		
		
		#pragma omp parallel for
		for (int ipartition=0; ipartition<npartitions; ipartition++)
		{
			int m_p = p[ipartition+1] - p[ipartition];
		
			pGamma[ipartition] = oski_CreateMatCSR (prow_offs[ipartition], jp[ipartition], vp[ipartition], m_p, Ncols, SHARE_INPUTMAT,
			                       1, INDEX_ZERO_BASED);

			oski_SetHintMatMult (pGamma[ipartition], OP_NORMAL, 
				                 1.0, SYMBOLIC_MULTIVEC,
				                 0.0, SYMBOLIC_MULTIVEC, 100);

			//#pragma omp barrier
			oski_TuneMat (pGamma[ipartition]);
		
			py[ipartition] = oski_CreateMultiVecView ((DATA_T*)kgridded+p[ipartition]*2, m_p, 2*nch, LAYOUT_ROWMAJ, 2*nch);
		}
		
		return 0;
	}
	
	/*// constructor:
	SpM_CSR_t(FILE* implementationfile)
	{
		fread(&Nrows, sizeof(int), 1, implementationfile);
		fread(&Ncols, sizeof(int), 1, implementationfile);
		row_offs = (int*) calloc (Nrows+1, sizeof(int));
		fread(&row_offs, sizeof(int), Nrows+1, implementationfile);
		nnz = row_offs[Nrows];
		col_idcs = (int*) calloc (nnz, sizeof(int));
		fread(&col_idcs, sizeof(int), row_offs[Nrows], implementationfile);
		values = (int*) calloc (nnz, sizeof(int));
		fread(&values, sizeof(int), row_offs[Nrows], implementationfile);
		
		//unify with previous constructor
		//create_Gamma();
	}*/
	
	
	int read_from_file(FILE* fid)
	{
		fread(&Nrows, sizeof(int), 1, fid);
		fread(&Ncols, sizeof(int), 1, fid);
	
		row_offs = (int*) calloc (Nrows+1, sizeof(int));
		fread(row_offs, Nrows+1, sizeof(int), fid);
		
		nnz = row_offs[Nrows];
		
		col_idcs = (int*) calloc (nnz, sizeof(int));
		fread(col_idcs, nnz, sizeof(int), fid);

		values = (DATA_T*) calloc(nnz, sizeof(DATA_T));
		fread(values, nnz, sizeof(DATA_T), fid);
		
		return 0;
	}
	
	int write_to_file(FILE* fid)
	{
		fwrite(&Nrows, sizeof(int), 1, fid);
		fwrite(&Ncols, sizeof(int), 1, fid);
	
		fwrite(row_offs, sizeof(int), Nrows+1, fid);
		fwrite(col_idcs, sizeof(int), nnz, fid);

		fwrite(values, sizeof(DATA_T), nnz, fid);
		
		return 0;
	}
	
	
	// destructor:
	int Free()
	{
		oski_DestroyMat(Gamma);
		oski_DestroyVecView(x);
		for (int ipartition=0; ipartition<npartitions; ipartition++) {
			oski_DestroyMat(pGamma[ipartition]);
			oski_DestroyVecView(py[ipartition]);
			free(prow_offs[ipartition]);
		}
		free(prow_offs);
		free(row_offs);
		free(col_idcs);
		free(values);
		free(pGamma);
		free(py);
		//free oski wisdom?
		return 0;
	}
};
