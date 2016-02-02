  function [xs, info] = pl_pcg_qs_ls(x, G, data, dercurv, R, varargin)
%|function [xs, info] = pl_pcg_qs_ls(x, G, data, dercurv, R, varargin)
%|
%| Unconstrained generic penalized-likelihood minimization,
%| for arbitrary negative log-likelihood with convex non-quadratic penalty,
%| via preconditioned conjugate gradient algorithm
%| with quadratic surrogate based line search.
%| cost(x) = -log p(data|x) + R(x)
%|
%| in
%|	x	[np,1]		initial estimate
%|	G	[nd,np]		system matrix
%|	data	{cell}		whatever data is needed for the likelihood
%|	dercurv	function_handle function returning derivatives and curvatures
%|				of negative log-likeihood via:
%|				[deriv curv] = dercurv(data, G*x, curvtype)
%|	R			penalty object (see Robject.m)
%|
%| options (name / value pairs)
%|	niter	?		# total iterations
%|	isave	[]		list of iterations to archive
%|					(default: [] 'last')
%|	stepper	?		method for step-size line search
%|					default: {'qs', 3}
%|	precon	[np,np]		preconditioner, matrix | object; default: 1
%|	userfun			user defined function handle (see default below)
%|	curvtype		type of curvature, default 'pc'
%|
%| out
%|	xs	[np,nsave]	estimates each (saved) iteration
%|	info	[niter+1,?]	userfun output. default is: gamma, step, time
%|
%| Copyright 2004-2-1, Jeff Fessler, University of Michigan

if nargin == 1 && streq(x, 'test')
	run_mfile_local('qpwls_qn_test')
return
end
if nargin < 5, help(mfilename), error args, end
if ~isa(dercurv, 'function_handle'), error 'dercurv not function handle?', end

arg.stepper = {};
arg.niter = 1;
arg.isave = [];
arg.precon = 1;
arg.userfun = @userfun_default;
arg.curvtype = 'pc';

arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);

if isempty(arg.stepper)
	arg.stepper = {'qs', 3}; % quad surr with this # of subiterations
end

xs = zeros(length(x), length(arg.isave));
if any(arg.isave == 0)
	xs(:, arg.isave == 0) = x;
end

%info = zeros(arg.niter,?); % trick: do not initialize since size may change

C1 = R.C1; % instantiate object
Rdercurv = R.dercurv;

%
% initialize projections
%
Gx = G * x;
C1x = C1 * x;

cpu tic
oldinprod = 0;

%
% iterate
%
warned.dir = 0;
warned.step = 0;
for iter=1:arg.niter
	ticker(mfilename, iter, arg.niter)

	%
	% gradient of cost function
	%
	[hderiv hcurv] = feval(dercurv, data, Gx, arg.curvtype);
	[pderiv pcurv] = feval(Rdercurv, R, C1x);
	grad = G' * hderiv + C1' * pderiv;

	%
	% preconditioned gradient
	%
	pregrad = arg.precon * grad;

	%
	% direction
	%
	newinprod = grad' * pregrad;
	if oldinprod == 0
		ddir = -pregrad;
		gamma = 0;
	else
		gamma = newinprod / oldinprod;	% Fletcher-Reeves
%		gamma = (newinprod - oldgrad' * pregrad) / oldinprod;
		ddir = -pregrad + gamma * ddir;
	end
%	oldgrad = grad;
	oldinprod = newinprod;

	% check if descent direction
	if ddir' * grad > 0
		if ~warned.dir
			warned.dir = 1;
			warning 'wrong direction so resetting'
			printf('<ddir,grad>=%g, |ddir|=%g, |grad|=%g', ...
				ddir' * grad, norm(ddir), norm(grad))
%			keyboard
		end

		% reset
		ddir = -pregrad;
		oldinprod = 0;
	end


	%
	% step size in search direction
	%
	Gdir = G * ddir;
	C1dir = C1 * ddir;

	%
	% multiple steps based on quadratic surrogates
	%
	if streq(arg.stepper{1}, 'qs')
		nsub = arg.stepper{2};
		step = 0;
		for is=1:nsub
			if step ~= 0
				[hderiv hcurv] = feval(dercurv, data, ...
						Gx + step * Gdir, arg.curvtype);
				[pderiv pcurv] = feval(Rdercurv, R, ...
							C1x + step * C1dir);
			end
			denom = (Gdir.^2)' * hcurv + (C1dir.^2)' * pcurv;
			numer = Gdir' * hderiv + C1dir' * pderiv;
			if denom == 0
				warning 'found exact solution???  step=0 now!?'
				step = 0;
			else
				step = step - numer / denom;
			end

			if step < 0
				if ~warned.step
					warning 'downhill step?'
					printf('iter=%d step=%g', iter, step)
%					keyboard
				end
			end
		end
	end

	%
	% update
	%
	Gx = Gx + step * Gdir;
	C1x = C1x + step * C1dir;
	x = x + step * ddir;

	if any(arg.isave == iter)
		xs(:, arg.isave == iter) = x;
	end

	info(1+iter,:) = feval(arg.userfun);
end


% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default
out = evalin('caller', '[gamma step cpu(''toc'')]');
