% qpwls_art2_vs_sps.m
% compare ART2 and SPS algorithms for QPWLS problem
% Copyright June 2000, Jeff Fessler, The University of Michigan

%
% generate data
%
if ~isvar('yi'), disp 'setup for qpwls_art2_vs_sps'
	ig = image_geom('nx', 64, 'ny', 60, 'dx', 1);
	ig.mask = ig.circ > 0;
	sg = sino_geom('par', 'nb', ig.nx, 'na', round(0.6 * ig.nx / 2)*2, ...
		'dr', 1);
	em_wls_test_setup
	W = spdiag(double(wi(:)), 'nowarn');
prompt
end

if ~isvar('R'), disp 'make R'
	f.l2b = -2;
	R = Rbuild('leak', ig.mask, 4, 2^f.l2b, 'quad', inf, 0);
	C = R.C; % need a sparse matrix for ART2!
%	R = Robject(ig.mask, 'type_denom', 'matlab', 'beta', 2^f.l2b);
%	C = R.C(:,:);
%	C = sqrt(2^f.l2b) * speye(length(x));
%	C = sqrt(2^f.l2b) * Csparse('maskleak', ig.mask, 1);
prompt
end


%
% ART2
%
if ~isvar('xart2'), disp 'do art2'
	f.niter = 16;
	xinit = xfbp;	% trick: ART needs the negatives! (range(A'))

	elist = [2 1 0 -1 -2];	% 2 bad, 0 great, -2 bad, -1 ok
for ie=1:length(elist)
	f.eps = 10^elist(ie);
	xart2 = qpwls_art2(double(xinit(ig.mask)), G', ...
		double(wi), double(yi), C', f.eps, f.niter);
	xart2 = max(xart2,0);
	xart2 = ig.embed(xart2);
	im clf, im(xart2, 'Matlab QPWLS-ART2 iterations')
	clear yt
	xart{ie} = xart2;
end
prompt
end

%
% SPS
% fix: replace with incremental...
%
if ~isvar('xsps'), disp 'do sps'

	x = max(xinit, 0);
	xsps = pwls_sps_os(double(x(ig.mask)), double(yi), double(wi), ...
		G, R, f.niter, inf, [], [], 1, 0);
	xsps = ig.embed(xsps);

	im clf, im(xsps, 'Matlab QPWLS-SPS iterations')
prompt
end

% make figure for various epsilons - results are very sensitive!
if 1
	t0 = pwls_cost(xsps, G, W, double(yi(:)), R, ig.mask);
	for ie=1:length(elist)
		t1(:,ie) = pwls_cost(xart{ie}, G, W, double(yi(:)), R, ig.mask) - t0(1);
	end
	t0 = t0 - t0(1);
	plot(0:f.niter-1, t0, '-o', 0:f.niter-1, t1, '-x')
	xlabel iteration, ylabel '\Phi change', legend('SPS', 'ART2', 1)
	title('QPWLS: SPS vs ART2')
	hold on
	for ie=1:length(elist)
		t = sprintf('10^{%g}', elist(ie));
		text(2, 10+t1(3,ie), t)
	end
	hold off

%	savefig 'fig_qpwls_art2_vs_sps'
return
end

	im clf, im(221, xart2, 'ART2 matlab')
	im(222, xsps, 'SPS matlab')
%	im(223, (xart2-xmat)/max(xmat(:)), '(aspire-matlab)/max(matlab)')

	t1 = pwls_cost(xart2, G, W, double(yi(:)), R, ig.mask);
	t2 = pwls_cost(xsps, G, W, double(yi(:)), R, ig.mask);

	subplot(223)
	plot(0:f.niter-1, t1-t1(1), '-o', 0:f.niter-1, t2-t1(1), '-x')
	xlabel iteration, ylabel '\Phi change', legend('ART2', 'SPS', 1)
	title('QPWLS, ART2 vs SPS')
%	axisy(-4.6e4, -4.4e4)
%	axisy(-4.5e4, -4.2e4)
