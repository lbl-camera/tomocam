 function ob = Gdsft(om, Nd, varargin)
%function ob = Gdsft(om, Nd, varargin)
% Construct Gdsft object, which computes (nonuniform) FT samples
% of signals with dimensions [(Nd)] exactly, e.g., for testing Gnufft
% For faster computation, use Gnufft instead.
% See Gdsft_test.m for example usage.
%
% in
%	om	[M,D]		frequency locations (radians / sample)
%	Nd	[1,D]		signal dimensions
%
% options
%	mask	[(Nd)]		logical support array
%	n_shift	[1,D]		see nufft_init
%	nthread	[]		# of processor threads
% out
%	ob			Fatrix object
%
% Copyright 2005-7-22, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(om, 'test'), Gdsft_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

% defaults
arg.om_t = double(om)'; % [D,M], as required by dtft_mex
arg.Nd = Nd;
arg.ndim = length(arg.Nd);
if size(arg.om_t,1) ~= arg.ndim, error 'dimension mismatch', end
arg.mask = [];
arg.n_shift = zeros(1, arg.ndim);
arg.nthread = 1;

% options
arg = vararg_pair(arg, varargin);

if isempty(arg.mask)
	arg.mask = true([arg.Nd 1]); % [(Nd)]
end
arg.np = sum(arg.mask(:));
arg.dim = [size(arg.om_t,2) arg.np]; % M x np

%
% initialize
%
arg = Gdsft_init(arg);

%
% build Fatrix object
%
ob = Fatrix(arg.dim, arg, 'forw', @Gdsft_forw, 'back', @Gdsft_back, ...
	'caller', mfilename);
%	'gram', @Gdsft_gram,  ...


%
% Gdsft_init()
%
function arg = Gdsft_init(arg)

if ~isempty(arg.n_shift)
% fix: - ?
	arg.phasor = exp(1i * (arg.om_t' * arg.n_shift(:))); % [M,1]
	arg.phasor = diag_sp(arg.phasor); % trick: to handle multiples
else
	arg.phasor = 1;
end


%
% Gdsft_forw(): y = G * x
% in:
%	x	[np,L] or [(Nd),L]
% out:
%	y	[M,L]
%
function y = Gdsft_forw(arg, x)

if size(x,1) == arg.np
	x = embed(x, arg.mask);	% [(Nd),(L)]
end
y = jf_mex('dtft,forward', arg.om_t, double(x), int32(arg.nthread));
y = arg.phasor * y;


%
% Gdsft_back(): x = G' * y
% in:
%	y	[M,L]
% out:
%	x	[np,L]
%
function x = Gdsft_back(arg, y)

y = arg.phasor' * y;
if isreal(y)
	y = complexify(y);
end
x = jf_mex('dtft,adjoint', arg.om_t, y, int32(arg.Nd), int32(arg.nthread));
x = reshape(x, prod(arg.Nd), []);
x = x(arg.mask(:),:);


%
% Gdsft_gram()
%
function [T, reuse] = Gdsft_gram(ob, W, reuse)
% T = dsft_gram(ob, W);
error 'not done'
