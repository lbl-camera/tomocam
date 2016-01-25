 function x = wls_simplex(A, y, Wh, x, varargin)
%function x = wls_simplex(A, y, Wh, x, [options])
%| min_x || Wh * (A x - y) ||
%| subject to simplex constraint: 0 <= x <= 1 and sum(x) = 1
%|
%| based on:
%| x = lsqlin(C,d,A,b,Aeq,beq) solves the least-squares
%| (with equality constraints) problem:
%| min_x 0.5*(norm(C*x-d)).^2 subject to A*x <= b and Aeq*x = beq
%| x = lsqlin(C,d,A,b,Aeq,beq,LB,UB) defines a set of lower and upper
%| bounds on the design variables, x, so that LB <= x <= UB.
%|
%| option
%|	'inprodv'	if (row vector) provided, require 1 = inprodv * x
%|
%| Copyright 2006-1-1, Jeff Fessler, University of Michigan

if nargin < 2, help(mfilename), error(mfilename), end

arg.inprodv = [];
arg = vararg_pair(arg, varargin);

n = ncol(A);

if ~isequal([nrow(A) 1], size(y))
	fail 'y must be [nrow(A) 1]'
end

if ~isvar('Wh') || isempty(Wh)
	Wh = 1;
end

if ~isvar('x') || isempty(x)
	x = ones(n,1) / n;
else
	if ~isequal([n 1], size(x))
		fail 'x must be [n 1]'
	end
	x = max(x, 0);
	x = x / sum(x);
end

if isempty(arg.inprodv)
	Aeq = ones(1,n);
	Beq = 1;
else
	if ~isequal([1 n], size(arg.inprodv))
		fail 'inprodv must be [1 n]'
	end

	if all(arg.inprodv > 1)
		minmax(arg.inprodv)
		fail '<inprodv, x> = 1 infeasibile'
	end

	Aeq = [ones(1,n); arg.inprodv];
	Beq = [1; 1];
end

arg.maxiter = 400;
opt = optimset('largescale', 'off', 'display', 'off', 'maxiter', arg.maxiter);

[x resnorm residual exitflag output lambda] = ...
	lsqlin(Wh * A, Wh * y, [], [], Aeq, Beq, zeros(1,n), ones(1,n), x, opt);

if exitflag == 0
	warn('lsqlin exitflag=%d, may need more iterations', exitflag)
elseif exitflag ~= 1
	printm('lsqlin exitflag=%d', exitflag)
	keyboard
end

if isempty(resnorm)
	fail 'inconsistent input bounds'
end

% check for significant negative values
if any(x < -eps)
	minmax(x)
	warn 'x < 0'
%	keyboard
	x = max(x,0);
	x = x / sum(x);
end

% purge negligible negative values
if any(x < 0)
	printm('zeroing %d of %d tiny negatives', sum(x < 0), length(x))
	x = max(x,0);
	x = x / sum(x);
end

if any(x > 1)
	fail 'bug: x > 1'
end
