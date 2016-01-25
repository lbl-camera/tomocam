  function ob = Cdiffs(isize, varargin)
%|function C1 = Cdiffs(isize, [options])
%|
%| Construct C1 object that can compute C1 * x and the adjoint C1' * d
%| for a "differencing" matrix C for roughness penalty regularization.
%| This "stacks up" multiple Cdiff1() objects, e.g., using block_fatrix(),
%| for possible internal use by the roughness penalty objects.
%|
%| Caution: for large problem sizes, computing C1' * (C1 * x) will require
%| #offsets * #pixels intermediate memory to store C1*x, which may be too much.
%| Instead, one should compute \sum_{m=1}^M C1_m' * (C1_m * x).
%| So this object is provided mostly for completeness.
%|
%| in
%|	isize	[]		vector of object dimensions (N), e.g., [64 64]
%|
%| options
%|	'type_diff'		see Cdiff1.m (default: '' defers to Cdiff1)
%|	'offsets' [M]		offsets to "M" neighbors; see penalty_offsets()
%|	'order'	1 or 2		1st- or 2nd-order differences.  (default: 1)
%|	'mask'	[(N)]		logical support
%|
%| out
%|	C1	[*N * M, np]	Fatrix object, where np = sum(mask(:))
%|				also works on arrays: [(N)] -> [(N) M]
%|			or sparse matrix for type_diff == 'spmat'
%|			(but sparse version handles only vectors, not arrays)
%|
%| Copyright 2006-12-4, Jeff Fessler, University of Michigan

if nargin == 1 & streq(isize, 'test'), Cdiffs_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end
%if has_mex_jf, penalty_mex('help'), end

% option defaults
arg.type_diff = '';
arg.offsets = [];
arg.mask = [];
arg.order = 1;

% parse optional name/value pairs
arg = vararg_pair(arg, varargin);

% offsets to neighbors
arg.offsets = penalty_offsets(arg.offsets, isize);

MM = length(arg.offsets);

% sparse matrix case
if streq(arg.type_diff, 'spmat')
	ob = [];
	for mm=1:MM
		ob = [ob; Cdiff1(isize, 'type_diff', arg.type_diff, ...
			'offset', arg.offsets(mm), 'order', arg.order)];
	end
	if ~isempty(arg.mask)
		ob = ob(:,arg.mask(:));
	end

% typical object case
else
	arg.isize = isize;
	arg.Cc = cell(MM,1);
	for mm=1:MM
		arg.Cc{mm} = Cdiff1(isize, 'type_diff', arg.type_diff, ...
			'offset', arg.offsets(mm), 'order', arg.order);
	end

	if isempty(arg.mask)
		arg.mask = true(isize);
	end
	arg.np = sum(arg.mask(:));

	dim = [prod(isize)*MM arg.np];
	ob = Fatrix(dim, arg, 'caller', 'Cdiffs', ...
		'forw', @Cdiffs_forw, 'back', @Cdiffs_back, ...
                'power', @Cdiffs_power, 'abs', @Cdiffs_abs);

%	ob = block_fatrix(ob, 'type', 'col'); % old approach
end


%
% Cdiffs_forw(): y = G * x
%
function y = Cdiffs_forw(arg, x)

[x ei] = embed_in(x, arg.mask, arg.np); % [(N) *L]
LL = size(x, 1+length(arg.isize));

MM = length(arg.Cc);

y = zeros([prod(arg.isize)*LL MM]); % [*N * *L, M]
for mm=1:MM
	tmp = arg.Cc{mm} * x; % [(N) *L]
	y(:,mm) = tmp(:);
end

if LL > 1
	y = reshape(y, [prod(arg.isize) LL MM]); % [*N *L M]
	y = permute(y, [1 3 2]); % [*N M *L]
end
y = reshape(y, [arg.isize MM LL]); % [(N) M *L]

y = ei.shape(y); % [*N * M, (L)] or [(N) M (L)]


%
% Cdiffs_back(): x = G' * y
%
function x = Cdiffs_back(arg, y)

MM = length(arg.Cc);
[y eo] = embed_out(y, [arg.isize, MM]); % [(N) M *L]
LL = size(y, 2+length(arg.isize));

y = reshape(y, [prod(arg.isize) MM LL]); % [*N M *L]

if LL > 1
	y = permute(y, [1 3 2]); % [*N *L M]
	y = reshape(y, [prod(arg.isize)*LL MM]); % [*N * *L, M]
end

x = 0;
for mm=1:MM
	tmp = reshape(y(:,mm), [arg.isize LL]); % [(N) *L]
	tmp = arg.Cc{mm}' * tmp; % [(N) *L]
	x = x + tmp;
end

if LL > 1
	x = x .* repmat(arg.mask, [ones(1,ndims(arg.mask)) LL]);
else
	x = x .* arg.mask;
end
x = eo.shape(x, arg.mask, arg.np); % [*N (L)] or [(N) (L)]


%
% Cdiffs_abs()
%
function ob = Cdiffs_abs(ob)
MM = length(ob.arg.Cc);
Ca = cell(MM,1);
for mm=1:MM
	Ca{mm} = abs(ob.arg.Cc{mm});
end
ob.arg.Cc = Ca;


%
% Cdiffs_power()
% for C.^2
%
function ob = Cdiffs_power(ob, p)
MM = length(ob.arg.Cc);
Cp = cell(MM,1);
for mm=1:MM
	Cp{mm} = ob.arg.Cc{mm} .^ p;
end
ob.arg.Cc = Cp;


%
% Cdiffs_test()
%
function Cdiffs_test
ig = image_geom('nx', 8, 'ny', 6, 'dx', 1);
ig.mask = ig.circ > 0;

% x = ig.unitv;
rand('state', 0)
x = rand(ig.dim);

for order=1:2
	args = {'order', order, 'mask', ig.mask};
	C = Cdiffs(ig.dim, 'type_diff', 'ind', args{:});
	y1 = C * x;
	z1 = C' * y1;
	Fatrix_test_basic(C, ig.mask)

	Ci = C(:,:);

	types = {'def', 'ind', 'mex', 'sparse'};
	for it=1:length(types)
		C = Cdiffs(ig.dim, 'type_diff', types{it}, args{:});
		y2 = C * x;
		z2 = C' * y1;
		equivs(y1, y2)
		equivs(z1, z2)

		Ct = C(:,:);
		jf_equal(Ct, Ci)
	end

	im clf, im pl 1 2
	im(1, y1)
	im(2, z1)

	% test sparse matrix too
	Cs = Cdiffs(ig.dim, 'type_diff', 'sparse', args{:});
	Cz = Cdiffs(ig.dim, 'type_diff', 'spmat', args{:});
	Cf = Cs(:,:);
	jf_equal(Cf, Cz)

	% abs
	Ca = abs(Cs);
	Cf = Ca(:,:);
	jf_equal(Cf, abs(Cz))

	% abs
	Cp = Cs .^ 2;
	Cf = Cp(:,:);
	jf_equal(Cf, Cz .^ 2)
end
