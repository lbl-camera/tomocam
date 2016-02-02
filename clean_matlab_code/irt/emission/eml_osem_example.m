% eml_osem_example.m
%
% a complete example m-file illustrating E-ML-OSEM.
% showing faster "convergence" compared to E-ML-EM
%
% Copyright 1998-1, Jeff Fessler, The University of Michigan


%
% generate data
%
if ~isvar('yi'), printm 'data'
	f.count = 1e6;
	im clf, em_test_setup
prompt
end

% block object for block iterations
if ~isvar('Gb'), printm 'Gb'
	f.nblock = 5;
	Gb = Gblock(G, f.nblock, 0);
end

%
% ML-EM iterations
%
if ~isvar('xmlem'), printm 'E-ML-EM'
	f.niter = 16;
	xinit = ig.ones; % uniform initial image

	xmlem = eml_em(xinit(ig.mask), G, yi(:), ci(:), ri(:), [], f.niter);
	xmlem = ig.embed(xmlem);

	im clf, im pl 2 2, im(1, xmlem, 'ML-EM'), cbar
	printm('Done running ML-EM')
prompt
end


%
% OS-EM iterations
%
if ~isvar('xosem'), printm 'E-ML-OS-EM'
	xosem = eml_osem(xinit(ig.mask), Gb, yi, ci, ri, ...
		'niter', f.niter-1, 'precon', 'classic');
	xosem = ig.embed(xosem);
	im(2, xosem, 'E-ML-OS-EM classic'), cbar
prompt
end

if 0, disp 'E-ML-OS-EM fast'
	xfast = eml_osem(xinit(ig.mask), Gb, yi, ci, ri, ...
		'niter', f.niter-1, 'precon', 'fast');
	xfast = ig.embed(xfast);

	im(3, xfast, 'E-ML-OS-EM fast'), cbar
	im(4, xfast-xosem, 'fast-classic'), cbar
prompt
end


%
% E-ML-INC-EM-3 iterations
%
if ~isvar('xiem3'), disp 'E-ML-INC-EM'

	xiem1 = eml_inc_em(xinit(ig.mask), Gb, yi, ci, ri, ...
		'niter', f.niter, 'hds', 1);
	xiem1 = ig.embed(xiem1);
	im(3, xiem1, 'E-ML-INC-EM-1'), cbar

	xiem3 = eml_inc_em(xinit(ig.mask), Gb, yi, ci, ri, ...
		'niter', f.niter, 'hds', 3);
	xiem3 = ig.embed(xiem3);
	im(4, xiem3, 'E-ML-INC-EM-3'), cbar
	printm 'Done running E-ML-INC-EM'
prompt
end


% plot likelihood to show acceleration
if 1
	like.mlem = eql_obj(xmlem, G, yi(:), ci(:), ri(:), [], ig.mask);
	like.osem = eql_obj(xosem, G, yi(:), ci(:), ri(:), [], ig.mask);
	like.iem1 = eql_obj(xiem1, G, yi(:), ci(:), ri(:), [], ig.mask);
	like.iem3 = eql_obj(xiem3, G, yi(:), ci(:), ri(:), [], ig.mask);
	if im
		subplot(212)
		plot(	...
			0:f.niter-1, like.osem, 'c-+', ...
			0:f.niter-1, like.iem1, 'g*-', ...
			0:f.niter-1, like.iem3, 'm*-', ...
			0:f.niter-1, like.mlem, 'yx-', ...
			f.nblock * (0:f.niter-1), like.osem, 'ro')
		axisx(0, f.niter-1)
		legend('OS-EM', 'INC-EM-1', 'INC-EM-3', 'ML-EM', ...
			sprintf('OSEM/%d', f.nblock), 4)
		title 'ML-EM vs ML-OSEM convergence rate'
		xlabel iteration, ylabel likelihood
	end
prompt
end

% run regularized case for comparison
if 0 | ~isvar('R'), disp 'build R'
	f.l2b = -2;
	f.delta = 0.5;
	R = Robject(ig.mask, 'type_denom', 'matlab', ...
		'potential', 'hyper3', 'beta', 2^f.l2b, 'delta', f.delta);
prompt
end

if 0 | ~isvar('xh'), disp 'E-PL-OS-EMDP'
	xh = epl_os_emdp(xinit(ig.mask), Gb, yi, ci, ri, R, f.niter, 1e9, 1);
	xh = ig.embed(xh);
	im clf, im(xh, 'xh'), cbar
%prompt
end

if 1 & im % compare pics of ml-osem vs pl
	clim = [0 8];
	elim = [-1 1]*2.5;
	im(231, xtrue, clim, 'true'), cbar
	im(232, xosem(:,:,end), clim, 'ML-OSEM'), cbar
	im(233, xh(:,:,end), clim, 'PL-OS-EMDP'), cbar
	im(235, xtrue-xosem(:,:,end), elim), cbar
	title(sprintf('nrms err %g%%', 100*nrms(col(xosem(:,:,end)), xtrue(:))))
	im(236, xtrue-xh(:,:,end), elim), cbar
	title(sprintf('nrms err %g%%', 100*nrms(col(xh(:,:,end)), xtrue(:))))
end
