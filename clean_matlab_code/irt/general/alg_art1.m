  function xs = alg_art1(x, Gt, y, varargin)
%|function xs = alg_art1(x, Gt, y, [options])
%| classical ART algorithm, aka, Kaczmarz algorithm; tries to solve y=Ax
%|
%| in
%|	x	[np,1]		initial guess, possibly empty
%|	Caution: x must be in the range of G' for convergence!
%|	Gt	[np,nd]		*transpose* of system matrix
%|	y	[nb,na]		measurement
%|
%| option
%|	wi	[nb,na]		weights
%|	niter			# of iterations
%|	isave			default: [] 'last'
%|	pixmin
%|	pixmax
%|	gnorms	[nd,1]		see below
%|	eps
%|
%| out
%|	xs	[np,niter]	iterates
%|
%| Copyright 2006-4-2, Jeff Fessler, University of Michigan

if nargin == 1 && streq(x, 'test'), alg_art1_test, return, end
if nargin < 3, help(mfilename), error(mfilename), end
if isempty(x), x = zeros(nrow(G),1); end

% defaults
arg.niter = 1;
arg.isave = [];
arg.gnorms = [];
arg.eps = eps;
%arg.pixmax = inf;
%arg.pixmin = -inf;

% options 
arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);

[nb na] = size(y);
starts = subset_start(na);

%
% For WLS, premultiply y and postmultiply Gt by W^{1/2}
%
%Wh = spdiag(sqrt(wi(:)), 'nowarn');
%y = Wh * y(:);
%Gt = Gt * Wh;

%
% weighted row norms
%
if isempty(arg.gnorms)
	arg.gnorms = sum(Gt.^2);	% | e_i' G |^2
end

iglist = col(outer_sum(1:nb, (starts-1)*nb));
iglist = iglist(arg.gnorms(iglist) ~= 0);

gdenom = arg.gnorms + eps; % trick:

np = length(x);
xs = zeros(np, length(arg.isave));
if any(arg.isave == 0)
	xs(:, arg.isave == 0) = x;
end

ticker(mfilename, 1, arg.niter)

for iter=1:arg.niter
	ticker(mfilename, iter, arg.niter)

	for ii=iglist'
		g = Gt(:,ii);
		step = (y(ii) - g' * x) / gdenom(ii);
		x = x + g * step;

%		todo: try following approach to see if faster
%		[j ignore g] = find(Gt(:,ii));
%		xj = x(j);
%		step = (y(ii) - g' * xj) / gdenom(ii);
%		x(j) = xj - step * g;
	end

	if any(arg.isave == iter)
		xs(:, arg.isave == iter) = x;
	end
end
