 function ob = Gcascade(arg1, arg2, varargin)
%function ob = Gcascade(G1, G2, options)
%
% Construct Gcascade object, which is the cascade of two objects: G = G1 * G2.
%
% See Gcascade_test.m for example usage.
%
% in
%	arg1	matrix | Fatrix		object that can do "mtimes" and "size"
%	arg2	matrix | Fatrix		''
%
% options
%	chat				verbosity
%
% out
%	ob	G1 * G2
%
% Copyright 2005-7-21, Jeff Fessler, The University of Michigan

if nargin == 1 & streq(arg1, 'test'), Gcascade_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

% defaults
arg.chat = 0;
arg = vararg_pair(arg, varargin);

arg.G1 = arg1;
arg.G2 = arg2;

if isnumeric(arg1) && isscalar(arg1) % scalar * object
	dim = size(arg.G2);
else % G1 * G2
	if size(arg.G1, 2) ~= size(arg.G2, 1)
		error 'size mismatch'
	end
	dim = [size(arg.G1,1) size(arg.G2,2)];
end

%
% build Fatrix object
%
ob = Fatrix(dim, arg, 'caller', mfilename, ...
	'forw', @Gcascade_forw, 'back', @Gcascade_back, ...
	'power', @Gcascade_power);
%	'block_setup', @Gcascade_block_setup, ...
%	'mtimes_block', @Gcascade_mtimes_block, ...


%
% Gcascade_forw(): y = G * x
%
function y = Gcascade_forw(arg, x)

y = arg.G1 * (arg.G2 * x);


%
% Gcascade_back(): x = G' * y
%
function x = Gcascade_back(arg, y)

x = arg.G2' * (arg.G1' * y);


%
% Gcascade_power(): G.^p
%
function ob = Gcascade_power(ob, pow)
if isnumeric(ob.arg.G1) && isscalar(ob.arg.G1)
	ob.arg.G1 = ob.arg.G1 .^ pow;
	ob.arg.G2 = ob.arg.G2 .^ pow;
else
	error 'power defined only for cascade of scalar * object'
end


%
% Gcascade_test
%
function Gcascade_test
a = [1:4]';
b = [2:5]';
c = diag_sp(a);
d = diag_sp(b);
e = Gcascade(c, d);
%e .^ 2;
f = [3:6]';
equivs(e * f, a .* b .* f)
g = Gcascade(7, c); % scalar * object
equivs(g * f, 7 * a .* f)

h = c * d; % check Fatrix * Fatrix
equivs(h * f, e * f)

equivs(g.^3 * f, 7^3 * a.^3 .* f) % power
