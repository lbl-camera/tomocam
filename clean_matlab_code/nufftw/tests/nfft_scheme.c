// precomputation


/*
	l - imin
	ll_plain - igrid
	ix - nnz
	psi_index_g - col_idcs
	psi_index_f - row_offs
	psi - values
*/


for(j=0,ix=0,ix_old=0; j<ths->M_total; j++) {
	for (t = ths->d-1; t >= 0; t--) {
		nfft_uo(ths,j,&u[t],&o[t],t);
		l[t] = u[t];
		lj[t] = 0;
	} /* for(t) */
	t++;       

	for(l_L=0; l_L<lprod; l_L++, ix++) {
		for(t2=t; t2<ths->d; t2++) {
			phi_prod[t2+1] = phi_prod[t2]* PHI(ths->x[j*ths->d+t2] - ((R)l[t2])/((R)ths->n[t2]), t2);
			ll_plain[t2+1] = ll_plain[t2]*ths->n[t2] +(l[t2]+ths->n[t2])%ths->n[t2];
		} /* for(t2) */

		ths->psi_index_g[ix] = ll_plain[ths->d];
		ths->psi[ix] = phi_prod[ths->d];

		for(t = ths->d-1; (t > 0) && (l[t] == o[t]); t--) {
			l[t] = u[t];
			lj[t] = 0;
		} /* for(t) */

		l[t]++;
		lj[t]++;
	} /* for(l_L) */

	ths->psi_index_f[j] = ix-ix_old;
	ix_old = ix;
} /* for(j) */


#define PHI(x,d) ((R)((POW((R)(ths->m), K(2.0))\
    -POW((x)*ths->n[d],K(2.0))) > 0)? \
    SINH(ths->b[d] * SQRT(POW((R)(ths->m),K(2.0)) - \
    POW((x)*ths->n[d],K(2.0))))/(KPI*SQRT(POW((R)(ths->m),K(2.0)) - \
    POW((x)*ths->n[d],K(2.0)))) : (((POW((R)(ths->m),K(2.0)) - \
    POW((x)*ths->n[d],K(2.0))) < 0)? SIN(ths->b[d] * \
    SQRT(POW(ths->n[d]*(x),K(2.0)) - POW((R)(ths->m), K(2.0)))) / \
    (KPI*SQRT(POW(ths->n[d]*(x),K(2.0)) - POW((R)(ths->m),K(2.0)))):K(1.0)))

R PHI(x,d)
{
	R val;
	val = ((R)((POW((R)(ths->m), K(2.0)) - POW((x)*ths->n[d],K(2.0))) > 0) ?
	SINH(ths->b[d] * SQRT(POW((R)(ths->m),K(2.0)) - POW((x)*ths->n[d],K(2.0))))/(KPI*SQRT(POW((R)(ths->m),K(2.0)) - POW((x)*ths->n[d],K(2.0)))) :
	(((POW((R)(ths->m),K(2.0)) - POW((x)*ths->n[d],K(2.0))) < 0) ?
		SIN(ths->b[d] * SQRT(POW(ths->n[d]*(x),K(2.0)) - POW((R)(ths->m), K(2.0)))) / (KPI*SQRT(POW(ths->n[d]*(x),K(2.0)) - POW((R)(ths->m),K(2.0)))) :
		K(1.0)))
	return val;
}

/** computes 2m+2 indices for the matrix B
 */
static inline void nfft_uo(const nfft_plan *ths, const int j, int *up, int *op,
  const int act_dim)
{
  const R xj = ths->x[j * ths->d + act_dim];
  int c = LRINT(FLOOR(xj * ths->n[act_dim]));

  (*up) = c - (ths->m);
  (*op) = c + 1 + (ths->m);
}



//adjoint


{
	// convolution
	memset(g,0,n_total*sizeof(C));

	int k;
	int lprod, lprod_m1;
	for(int t = 0, lprod = 1; t < d; t++)
		lprod *= 2 * m + 2;
	lprod_m1 = lprod / (2 * m + 2);

	#pragma omp parallel for default(shared) private(k)
	for (k = 0; k < M; k++) {
		int l;
		int j = (nfft_flags & NFFT_SORT_NODES) ? index_x[2*k+1] : k;

		for (l = 0; l < lprod; l++) {
			C val = psi[j * lprod + l] * f[j];
			C *gref = g + psi_index_g[j * lprod + l];
			R *gref_real = (R*) gref;

			#pragma omp atomic
			gref_real[0] += creal(val);

			#pragma omp atomic
			gref_real[1] += cimag(val);
		}
	}


	// fft
	fftw(g2->g1,backward);


	// deapodization
	memset(f_hat,0,ths->N_total*sizeof(C));

	#pragma omp parallel for default(shared) private(k_L)
	for (k_L = 0; k_L < ths->N_total; k_L++) {
		int kp[ths->d];                       /**< multi index (simple)           */ //0..N-1
		int k[ths->d];                        /**< multi index in g_hat           */
		int ks[ths->d];                       /**< multi index in f_hat, c_phi_inv*/
		R c_phi_inv_k_val = K(1.0);
		int k_plain_val = 0;
		int ks_plain_val = 0;
		int t;
		int k_temp = k_L;

		for (t = ths->d-1; t >= 0; t--) {
			kp[t] = k_temp % ths->N[t];
			if (kp[t] >= ths->N[t]/2)
				k[t] = ths->n[t] - ths->N[t] + kp[t];
			else
				k[t] = kp[t];
			ks[t] = (kp[t] + ths->N[t]/2) % ths->N[t];
			k_temp /= ths->N[t];
		}

		for (t = 0; t < ths->d; t++) {
			c_phi_inv_k_val *= ths->c_phi_inv[t][ks[t]];
			ks_plain_val = ks_plain_val*ths->N[t] + ks[t];
			k_plain_val = k_plain_val*ths->n[t] + k[t];
		}

		f_hat[ks_plain_val] = g_hat[k_plain_val] * c_phi_inv_k_val;
	} /* for(k_L) */
}




//forward


// deapodization
for (k_L = 0; k_L < ths->N_total; k_L++) {
	int kp[ths->d];                       /**< multi index (simple)           */ //0..N-1
	int k[ths->d];                        /**< multi index in g_hat           */
	int ks[ths->d];                       /**< multi index in f_hat, c_phi_inv*/
	R c_phi_inv_k_val = K(1.0);
	int k_plain_val = 0;
	int ks_plain_val = 0;
	int t;
	int k_temp = k_L;

	for (t = ths->d-1; t >= 0; t--) {
		kp[t] = k_temp % ths->N[t];
		if (kp[t] >= ths->N[t]/2)
			k[t] = ths->n[t] - ths->N[t] + kp[t];
		else
			k[t] = kp[t];
		ks[t] = (kp[t] + ths->N[t]/2) % ths->N[t];
		k_temp /= ths->N[t];
	}

	for (t = 0; t < ths->d; t++) {
		c_phi_inv_k_val *= ths->c_phi_inv[t][ks[t]];
		ks_plain_val = ks_plain_val*ths->N[t] + ks[t];
		k_plain_val = k_plain_val*ths->n[t] + k[t];
	}

	g_hat[k_plain_val] = f_hat[ks_plain_val] * c_phi_inv_k_val;
} /* for(k_L) */



//fft


//convolution
for (k = 0; k < ths->M_total; k++) {
	int l;
	int j = (ths->nfft_flags & NFFT_SORT_NODES) ? ths->index_x[2*k+1] : k;
	ths->f[j] = K(0.0);
	for (l = 0; l < lprod; l++)
		ths->f[j] += ths->psi[j*lprod+l] * ths->g[ths->psi_index_g[j*lprod+l]];
}
    
