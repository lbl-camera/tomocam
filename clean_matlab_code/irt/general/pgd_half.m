 function [xs, info] = pgd_half(x, data, costgrad, M, niter)
%|function [xs, info] = pgd_half(x, data, costgrad, M, niter)
%|
%| Generic unconstrained minimization
%| via preconditioned gradient descent algorithm with step-halfing
%| (practical only when the cost function is fast to compute)
%| in
%|	x	[np,1]		initial estimate
%|	data	{cell}		whatever data is needed for the cost function
%|	costgrad {function_handle} function returning cost function and gradient via:
%|					[costf grad] = costgrad(x, data)
%|	M	[np,np]		preconditioner (use "1" if none) matrix | object
%|	niter			# total iterations
%| out
%|	xs	[np,niter]	estimates each iteration
%|	info	[niter, 2]	step, time
%|
%| Copyright 2005-1-10, Anastasia Yendiki, University of Michigan

if nargin < 5, help(mfilename), error args, end
if ~isa(costgrad, 'function_handle'), error 'costgrad not function handle?', end

if isempty(M), M = 1; end

xs = zeros(numel(x), niter);
xs(:,1) = x(:);

info = zeros(niter,2);

tic

for iter=2:niter
	ticker(mfilename, iter, niter)

	%
	% gradient of cost function
	%
	[costf, grad] = feval(costgrad, x, data);

        %
        % preconditioned gradient
        %
        pregrad = M * grad;

	%
	% direction
	%
	ddir = -reshape(pregrad, size(x));

	step = 1;
	xnew = x + step*ddir;
	while (feval(costgrad, xnew, data) > costf)
		step = step/2;
		xnew = x + step*ddir;
	end

	%
	% update
	%
	x = xnew;
	xs(:,iter) = x(:);

	info(iter,:) = [step toc];
end
