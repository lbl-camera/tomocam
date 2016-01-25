% restore_sparse.m
% Example of image restoration with a l_1 sparsity constraint.
%
% Copyright 2005-4-22, Jeff Fessler, The University of Michigan

if ~isvar('xtrue'),	disp 'read image'
	rand('state', 0)
	xtrue = rand(70, 64) > 0.995;
	[nx ny] = size(xtrue);
	im clf, pl = 230;
	im(pl+1, xtrue, 'xtrue'), cbar
end

if ~isvar('psf'), disp 'psf of blur'
	psf = gaussian_kernel(3, 5); 
	psf = psf * psf';
end

if ~isvar('G'), disp 'G - system matrix'
	mask = true(nx, ny);
	G = Gblur(mask, 'psf', psf);
end

if ~isvar('yi'), disp 'yi - noisy data'
	y0 = G * xtrue;

	randn('state', 0)
	estd = 0.01;
	yi = y0 + estd * randn(size(y0));
	im(pl+2, y0, 'y0'), cbar
	im(pl+3, yi, 'yi'), cbar
end

if ~isvar('xqpls'), disp 'quadratic regularization case - straw man'
	f.l2b_q = -1;
	f.niter = 20;
	Rq = Robject(mask, 'type_denom', 'matlab', ...
		'potential', 'quad', 'beta', 2^f.l2b_q);

	xinit = yi;
	xqpls = pwls_sps_os(xinit(:), yi(:), [], G, Rq, ...
			f.niter, inf, [], [], 1);
	xqpls = embed(xqpls, mask);
	im(pl+4, xqpls, 'QPLS'), cbar
end

if ~isvar('xnpls') | 1, disp 'l1 regularization for sparsity'
	f.l2b_n = 2;
	Rn = Robject(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'potential', 'hyper3', 'beta', 2^f.l2b_n, 'delta', 0.01);
	xinit = yi;
	xnpls = pwls_sps_os(xinit(:), yi(:), [], G, Rn, ...
		3*f.niter, inf, [], [], 1);
	xnpls = embed(xnpls, mask);
	im(pl+5, xnpls, 'NPLS (L1, hyperbola)'), cbar
end

if 1, disp 'cauchy regularization - nonconvex for even more sparsity!'
	Rn = Robject(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'potential', 'cauchy', 'beta', 2^f.l2b_n, 'delta', 0.01);

	xinit = xnpls(:,:,end);
	xc = pwls_sps_os(xinit(:), yi(:), [], G, Rn, ...
		3*f.niter, inf, [], [], 1);
	xc = embed(xc, mask);
	im(pl+6, xc, 'NPLS (cauchy)'), cbar
end
