% restore_example2.m
% Example of edge-preserving image restoration
% using Gblur object for system model (shift-invariant blur)
% and using Robject.m to form nonquadratic penalty function.
%
% Copyright 2005-5-18, Jeff Fessler, The University of Michigan

if ~isvar('xtrue'), disp 'read image'
	xtrue = double(imread('cameraman.tif'))';
	[nx ny] = size(xtrue);
end

% PSF from figueiredo:05:abo
if ~isvar('psf'), disp 'psf'
	psf = -7:7;
	psf = ndgrid(-7:7, -7:7);
	psf = 1 ./ (1 + psf.^2 + (psf').^2);
	psf = psf / sum(psf(:)); % normalize to unity DC response
	im clf, im(221, psf, 'psf')
end

if ~isvar('G')
	G = Gblur(true(nx, ny), 'psf', psf);
	mask = G.arg.mask;
end

if ~isvar('yi')
	y0 = conv2(xtrue, psf, 'same');

	randn('state', 0)
	estd = sqrt(2);
	yi = y0 + estd * randn(size(y0));

	clim = [0 255];
	im(222, yi, 'yi', clim)
end

if ~isvar('xnpls') | 1
%	f.l2b_n = 1.5; % cauchy
	f.l2b_n = -6; % 
	f.delta = 0.1;
	Rn = Robject(mask, 'type_denom', 'matlab', ...
		'potential', 'hyper3', 'beta', 2^f.l2b_n, 'delta', f.delta);
%		'potential', 'cauchy', 'beta', 2^f.l2b_n, 'delta', 10);

	f.niter = 70;
	xinit = yi(mask);
	xnpls = pwls_sps_os(xinit, yi(:), [], G, Rn, ...
		f.niter, [0 inf], [], [], 1);

	snr_improve = 10 * log10(sum(col(yi-xtrue).^2) ./ ...
		sum((xnpls-xtrue(:)*ones(1,f.niter)).^2));
	if im
		subplot(224), plot(0:f.niter-1, snr_improve, '-o')
	end

	xnpls = embed(xnpls, mask);
	im(223, xnpls(:,:,end), 'xnpls', clim)
end
