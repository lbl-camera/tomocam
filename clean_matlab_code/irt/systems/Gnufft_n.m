 function ob = Gnufft_n(varargin)
%function ob = Gnufft([mask,] args)
% Construct Gnufft object, which computes nonunform FT samples
% of signals with dimensions [(Nd)] approximately via the NUFFT.
% For exact (but slow) computation, use Gdsft instead.
%
% The arguments (args) are simply a cell array of the all the arguments
% that will be passed to "nufft_init()" in the appropriate order.
% See Gnufft_test.m for example usage.
%
% Alternatively, the input can be a struct of the type returned by nufft_init().
%
% Basically, you create a system matrix object by calling:
%	G = Gnufft( ... )
% and then you can use it thereafter by typing commands like
%	y = G * x;
% which will auto-magically evaluate the DSFT samples.
% This is useful for iterative image reconstruction in MRI.
%
% Besides simple utilities like display, there are the following
% capabilities of this object:
%	y = G * x		forward operation
%	x = G' * y		adjoint operation
%	
% Optional arguments
%	mask		logical support array
%
% Copyright 2003-6-1, Jeff Fessler, The University of Michigan


arg.mask = varargin{1};
varargin = {varargin{2:end}};


arg.arg = varargin{1};
arg.st = nufft_init(arg.arg{:}); % initialize NUFFT structure

if ~isvar('arg.mask')
    'mask'
	arg.mask = true([arg.st.Nd 1]); % [(Nd)]
end

arg.dim = [nrow(arg.st.om) sum(arg.mask(:))]; % M x np

arg.new_mask = @Gnufft_new_mask;

arg.dim

%
% build Fatrix object
%
ob = Fatrix(arg.dim, arg, 'forw', @Gnufft_forw, 'back', @Gnufft_back, ...
	'gram', @Gnufft_gram, 'caller', mfilename);


%
% Gnufft_new_mask()
%
function G = Gnufft_new_mask(G, mask)
if ~isequal(size(mask), size(G.arg.mask))
	error 'new mask size does not match old size'
end
G.arg.mask = mask;
G.arg.dim = [nrow(G.arg.st.om) sum(G.arg.mask(:))]; % M x np
G.dim = G.arg.dim;


%
% Gnufft_forw(): y = G * x
%
function y = Gnufft_forw(arg, x)

if size(x,1) == arg.dim(2)	% [np,(L)]
	x = embed(x, arg.mask);	% [(Nd),(L)]
end
y = nufft(x, arg.st); % [M,(L)]


%
% Gnufft_back(): x = G' * y
% in:
%	y	[M,L]
% out:
%	x	[np,L]
%
function x = Gnufft_back(arg, y)

x = nufft_adj(y, arg.st); % [(Nd),L]

%Ld = size(y); Ld = Ld(2:end);
Ns = prod(arg.st.Nd);
%x = reshape(x, [Ns prod(Ld)]); % [*Nd,*L]
x = reshape(x, Ns, []); % [*Nd,*L]
x = x(arg.mask,:); % [np,*L]
%x = reshape(x, [Ns Ld]); % [np,(L)] % not needed if L can be scalar only!


%
% Gnufft_gram()
%
function [T, reuse] = Gnufft_gram(ob, W, reuse)
T = nufft_gram(ob, W);
