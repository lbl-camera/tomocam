% pwls_example.m
%
% example of how to use PWLS algorithms for PET reconstruction, such as PWLS-PCG
%
% Copyright 2003-11-30, Jeff Fessler, The University of Michigan


%
% generate data
%
if ~isvar('yi'), printm 'setup'
	em_wls_test_setup
%	wi = ones(size(wi)); warning 'uniform wi' % to test circulant precon
	W = diag_sp(wi(:));
prompt
end

%
% regularization
%
if ~isvar('R'), printm 'R'
	f.l2b = 9;
	f.delta = 1;

	if 1
		Rq = Robject(ig.mask, 'beta', 2^f.l2b);
		psf = qpwls_psf(G, Rq.C, 1, ig.mask);
		im(7, psf, 'PSF'), cbar
	end, clear Rq

	kappa = sqrt( (G' * wi(:)) ./ (G' * ones(size(wi(:)))) );
	kappa = ig.embed(kappa);
	im(8, kappa, 'kappa'), cbar

	R = Robject(kappa, 'type_denom', 'matlab', ...
		'potential', 'hyper3', 'beta', 2^f.l2b, 'delta', f.delta);

	clear xos xiot xcg
prompt
end

f.niter = 20;

if ~isvar('xinit')
	%xinit = ones(size(xtrue));		% uniform initial image
	%xinit = xfbp;				% FBP initial image
	xinit = imfilter(xfbp, double(psf));	% smoothed FBP initial image
	im(9, xinit, 'init'), cbar
prompt
end


if ~isvar('Gb'), printm 'do Gb'
	f.nsubset = 40;
	Gb = Gblock(G, f.nsubset);
prompt
end

%
% OS-SPS iterations (unconstrained)
%
if ~isvar('xos'), printm 'do os'
	xlim = [-inf inf]; % unconstrained
	[xos tim.os] = pwls_sps_os(xinit(ig.mask), yi, wi, Gb, R, ...
			1+f.niter, xlim, [], [], 1);
	xos = ig.embed(xos);
	im clf, im(xos, 'xos')
prompt
end

%
% PWLS-IOT iterations (unconstrained)
%
if 0 | ~isvar('xiot'), printm 'do iot'
%profile on
	[xiot, tim.iot] = pl_iot(xinit(ig.mask), Gb, {yi, wi}, R, ...
			'dercurv', 'wls', ...
			'riter', 1, ...
			'os', 5, ... % f.niter, ...
			'niter', f.niter, 'isave', 'all', ...
			'pixmin', xlim(1), 'pixmax', xlim(2), ...
			'chat', 0);
%profile report
	xiot = ig.embed(xiot);
	im clf, im(xiot, 'xiot')
	minmax(xiot-xos)
prompt
end


%
% CG iterations
%
if ~isvar('xcg'), printm 'xcg'
	[xcg, tim.cg] = pwls_pcg1(xinit(ig.mask), G, W, yi(:), R, ...
			'niter', f.niter, 'isave', 'all');
	xcg = ig.embed(xcg);
	im clf, im(xcg, 'CG')
prompt
end


%
% compare CG and OS-SPS
%
if 1, printm 'cost plots'
	cost.cg		= pwls_cost(xcg,	G, W, yi(:), R, ig.mask);
	cost.os		= pwls_cost(xos,	G, W, yi(:), R, ig.mask);
	cost.iot	= pwls_cost(xiot,	G, W, yi(:), R, ig.mask);
	ii = 0:f.niter;
	im clf
	subplot(211)
	plot(ii, cost.os, 'y-o', ii, cost.cg, 'g-x', ii, cost.iot, 'c-+')
	xlabel 'iteration', ylabel 'cost'
	legend('OS-SPS', 'CG', 'IOT')
	subplot(212)
	plot(tim.os, cost.os, 'y-o', tim.cg(:,3), cost.cg, 'g-x', ...
		tim.iot, cost.iot, 'c-+')
	xlabel 'time', ylabel 'cost'
	legend('OS-SPS', 'CG', 'IOT')
%	minmax(diff(tim.os))
%	minmax(diff(tim.cg))
end

if 0, printm 'images'
	clim = [0 6];
	im clf, im([xtrue, xfbp; xos(:,:,end), xiot(:,:,end); ...
		xcg(:,:,end), xinit], clim), cbar
end
