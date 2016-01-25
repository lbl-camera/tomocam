 function [xs, steps] = qpwls_psd(G, W, yy, x, C, M, niter)
%function [xs, steps] = qpwls_psd(G, W, yy, x, C, M, niter)
%
% quadratic penalized weighted least squares (QPWLS) via
% preconditioned steepest descent (PSD) algorithm
% cost(x) = (y-Gx)'W(y-Gx) / 2  + x'C'Cx / 2
% in
%	G	[nn,np]		system matrix
%	W	[nn,nn]		data weighting matrix
%	yy	[nn,1]		noisy data
%	x	[np,1]		initial estimate
%	C	[nc,np]		penalty 'derivatives' (R = \Half C'*C)
%	M	[np,np]		preconditioner (matrix or object)
%	niter			# total iterations
% out
%	xs	[np,niter]	estimates each iteration
%	steps	[niter,1]	step size each iteration
%
% Copyright Jun 2000, Jeff Fessler, The University of Michigan

if nargin < 3 | nargin > 7, help(mfilename), error(mfilename), end
np = ncol(G);

if ~isvar('C') | isempty(C), C = 0; end
if ~isvar('M'), M = []; end
if ~isvar('niter') | isempty(niter), niter = 2; end
if ~isvar('x') | isempty(x), x = zeros(np,1); end

yy = yy(:);
xs = zeros(np, niter);
xs(:,1) = x;

steps = zeros(niter,2);

%
% initialize projections
%
Gx = G * x;
Cx = C * x;

%
% iterate
%
for ii=2:niter
	%
	% (negative) gradient
	%
	grad = G' * (W * (yy-Gx)) - C' * Cx;

	%
	% preconditioned gradient: search direction
	%
	if ~isempty(M)
		ddir = M * grad;
	else
		ddir = grad;
	end

	Gdir = G * ddir;
	Cdir = C * ddir;

	% check if descent direction
	if ddir' * grad < 0
		warning('wrong direction')
		keyboard
	end

	%
	% step size in search direction
	%
	step = (ddir' * grad) / (Gdir'*(W*Gdir) + Cdir'*Cdir);
	steps(ii,1) = step;
	if step < 0
		warning('downhill?')
		keyboard
	end

	%
	% update
	%
	x	= x + step * ddir;
	Gx	= Gx  + step * Gdir;
	Cx	= Cx  + step * Cdir;
	xs(:,ii) = x;
end
