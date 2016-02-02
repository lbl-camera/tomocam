  function [cost, fit, reg] = pwls_cost(xs, G, W, yi, R, mask)
%|function [cost, fit, reg] = pwls_cost(xs, G, W, yi, R, mask)
%| compute PWLS cost for each column of x
%| in
%|	xs	[np,niter]	iterates
%|	G	[nd,np]		system matrix
%|	W	[np,np]		data weighting matrix, usually diag_sp(wi)
%|	yi	[nd,1]		data
%|	R			penalty object (see Robject.m)
%|				or just *sparse* C matrix!
%| out
%|	cost	[niter,1]	cost
%|
%| Copyright 2002-2-12, Jeff Fessler, University of Michigan

if nargin < 4, help(mfilename), error(mfilename), end

if ~isvar('R'), R = []; end

if isvar('mask') & ~isempty(mask)
	xs = reshapee(xs, prod(size(mask)), []);	% [(*N),niter]
	xs = xs(mask(:), :);				% [np,niter]
end

niter = size(xs,2);
reg = zeros(niter,1);

if isempty(R)
	warning 'empty R means no penalty'

elseif issparse(R) | isa(R, 'Fatrix')
	C = R; % trick!
	if size(C,2) == size(C,1)
		warning 'square C is quite unusual!?'
	end
	for ii=1:niter
		reg(ii) = sum(abs(C * xs(:,ii)).^2)/2;
	end

elseif isstruct(R) | isa(R, 'strum')
	for ii=1:niter
%		reg(ii) = sum(P.pot(P.wt, P.C * xs(:,ii)));
		reg(ii) = R.penal(R, xs(:,ii));
	end

else
	keyboard
	error 'bad R'
end

fit = zeros(niter,1);
for ii=1:niter
	resid = yi - G * xs(:,ii); % predicted measurements
	fit(ii) = resid' * (W * resid) / 2;
end

cost = fit + reg;
cost = reale(cost, 'warn'); % trick: x'*x is not always real for complex values
