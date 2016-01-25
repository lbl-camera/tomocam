% unwrap_example.m
% example of phase unwrapping using optimization transfer
% fix: UNDER DEVELOPMENT - NOT TESTED OR DEBUGGED!
% Copyright 2003-6-15, Jeff Fessler, The University of Michigan

if ~isvar('x.true')
%	x.max_diff = pi/2;
	x.max_diff = 8*pi/9;
	nx = 41; ny = 31;
	p.x = [1:nx]-(nx/2+1);
	p.y = [1:ny]-(ny/2+1);
	x.true = gaussian_kernel(nx/3, (nx-1)/2) ...
		* gaussian_kernel(nx/3, (ny-1)/2)';
	% create phase with certain maximum
	x.true = x.true * x.max_diff / ...
	max(max(col(abs(diff(x.true,1)))), max(col(abs(diff(x.true,2)))));

	% true magnitude
	rand('state', 0)
	mag = unifrnd(0.2, 1.0, nx, ny);	% random
	mag(1:round(nx/2),:) = 1;		% uniform
	im clf, im pl 2 4
	x.clim = [0 9*pi];
	im(1, x.true, 'x true', x.clim), cbar
	im(2, mag, 'mag'), cbar
end

if ~isvar('yi'),	disp 'simulate noisy data'
	f.sig = 0.3;
	randn('state', 0)
	yi = mag .* exp(1i * x.true) + f.sig * randn(size(x.true));
	x.u = unwrap(unwrap(angle(yi))')';
	x.u = 0.5 * (x.u + unwrap(unwrap(angle(yi)')'));

	im(3, abs(yi), '|y|'), cbar
	im(4, angle(yi), '\angle y'), cbar
%	im(4, phase(yi), '\angle y'), cbar
	im(5, x.u, 'matlab unwrap', x.clim), cbar
end

if 1 | ~isvar('x.sps'),	disp 'iterative estimation'
	f.l2b_q = -3;	% choosing this beta may be the hardest part!
	Rq = Robject(true(nx,ny), 'type_denom', 'matlab', ...
		'beta', 2^f.l2b_q);

	niter = 200;
	niter = 40;
	x.init = x.u;
%	x.init = angle(yi);
	x.sps = unwrap_sps(x.init(:), yi(:), Rq, niter);
	x.sps_end = reshape(x.sps(:,end), nx, ny);

	im(6, x.sps_end, 'fessler sps unwrap', x.clim), cbar
end

cost = abs(yi).^2 .* (1 - cos(x.sps_end - angle(yi)));
im(7, cost, 'data mismatch'), cbar

a = angle(yi(:));
cost = abs(yi(:)).^2' * (1 - cos(x.sps - a(:,ones(1,niter))));
cost = cost + Rq.penal(Rq, x.sps);
if im
	im subplot 8, plot(0:niter-1, cost, '-o')
	xlabel 'iteration', title 'cost'
end
