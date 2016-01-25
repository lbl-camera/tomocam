 function ob = Gdown(Nd, varargin)
%function ob = Gdown(Nd, varargin)
% Construct Gdown object, which performs downsampling.
% This is useful perhaps for iterative zooming methods.
% See Gdown_test.m for example usage.
%
% in
%	Nd	[1,D]		input signal dimensions
%
% options
%	down	1|2|3|...		down sampling factor
%	type	func|Gsparse|...	not done.  default: 'func'
% out
%	ob			Fatrix object
%
% Copyright 2006-8-25, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(Nd, 'test'), Gdown_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

% defaults
arg.idim = Nd;
arg.down = 1;
arg.type = 'func';

% options
arg = vararg_pair(arg, varargin);

arg.odim = arg.idim ./ arg.down;
tmp = round(arg.odim) * arg.down;
if any(tmp ~= arg.idim)
	error 'only integer multiple image size supported'
end

if length(arg.idim) > 2
	error 'only 2d implemented due to downsample2() limitation'
end

arg.mask = true(arg.idim);
arg.np = sum(arg.mask(:));
arg.dim = [prod(arg.odim) arg.np]; % nd x np

arg.up_scale = 1 / arg.down^length(arg.idim);

%
% build Fatrix object
%
ob = Fatrix(arg.dim, arg, ...
	'forw', @Gdown_func_forw, 'back', @Gdown_func_back, ...
	'caller', mfilename);


%
% Gdown_func_forw(): y = G * x
% in:
%	x	[np,L] or [(Nd),L]
% out:
%	y	[M,L]
%
function y = Gdown_func_forw(arg, x)

[x ei] = embed_in(x, arg.mask, arg.np);

LL = size(x, 1+length(arg.idim)); % *L
if LL == 1
	y = downsample2(x, arg.down);
else
	y = [];
	for ll=1:LL
		y(:,:,ll) = downsample2(x(:,:,ll), arg.down); % fix: generalize
	end
end

y = ei.shape(y);


%
% Gdown_func_back(): x = G' * y
% in:
%	y	[M,L]
% out:
%	x	[np,L]
%
function x = Gdown_func_back(arg, y)

[y eo] = embed_out(y, arg.odim);

LL = size(y, 1+length(arg.odim)); % *L
if LL == 1
	x = upsample_rep(y, arg.down);
else
	for ll=1:LL
		x(:,:,ll) = upsample_rep(y(:,:,ll), arg.down); % fix: generalize
	end
end
x = x * arg.up_scale;

x = eo.shape(x, arg.mask, arg.np);


%
% Gdown_test
%
function Gdown_test

dim = [4 6];
down = 2;
G = Gdown(dim, 'down', down);

x = reshape(1:prod(dim), dim);
y1 = downsample2(x, down);
y2 = G * x;
jf_equal(y1, y2)

y = y1;
x1 = upsample_rep(y, down) * G.arg.up_scale;
x2 = G' * y;
jf_equal(x1, x2)

mask = true(dim);
Fatrix_test_basic(G, mask)

test_adjoint(G);
