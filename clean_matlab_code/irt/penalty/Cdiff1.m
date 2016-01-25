  function ob = Cdiff1(isize, varargin)
%|function C1 = Cdiff1(isize, [options])
%|
%| Construct Cdiff1 object that can compute C1 * x and the adjoint C1' * d
%| for a basic "differencing" matrix C1 for roughness penalty regularization.
%| This object provides the finite difference in only a *single* direction,
%| so it is designed to be used internally by roughness penalty objects.
%| 
%| Cdiffs "stacks these up" which is more useful (if memory permits).
%|
%| For the usual 1st-order differences, each row of C1 is either all zeros
%| (near border) or all zeros except for a single +1 and -1 value.
%|
%| in
%|	isize	[]		vector of object dimensions (N), e.g., [64 64]
%|
%| options
%|	'type_diff'		method for performing differencing:
%|				'def' | '': try 'mex', otherwise 'ind' (default)
%|				'ind' - using matlab indexing
%|				'mex' - using penalty_mex file (preferable)
%|				'sparse' - using internal Gsparse object
%|				'spmat' - return sparse matrix itself
%|				'convn' - using matlab's convn, e.g. with [1 -1]
%|					(convn does not work; do not use)
%|	'offset' [int]		neighbor offset. (default: [1], left neighbor)
%|				Scalar, or [dx dy dz ...] with length(isize).
%|				If 0, then C1 = Identity.
%|	'order'	1 or 2		1st- or 2nd-order differences.  (default: 1)
%|
%| out
%|	C1	[*N *N]		Fatrix object
%|				or sparse matrix, if 'spmat' option.
%|				trick: also works [(N) (L)] -> [(N) (L)]
%|					or [*N (L)] -> [*N (L)]
%|
%| The name "C1" reminds us the elements are +/- 1 and 0 (for order=1).
%|
%| Copyright 2006-11-29, Jeff Fessler, University of Michigan

if nargin == 1 & streq(isize, 'test'), Cdiff1_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end

% required input argument(s)
arg.isize = isize;

% option defaults
arg.type_diff = 'def';
arg.offset = [1];
arg.order = 1;

% parse options
arg = vararg_pair(arg, varargin);

if length(arg.offset) > 1 % displacement vector instead of offset scalar
	if ~isequal(size(arg.offset), size(arg.isize)), error 'offset size', end
	arg.offset = sum(cumprod([1 arg.isize(1:end-1)]) .* arg.offset);
end

arg.is_abs = false;
arg.Cpower = 1; % start with C^1
arg.nn = prod(arg.isize);
arg.dim = [arg.nn arg.nn];

% default: try mex if available
if isempty(arg.type_diff) || streq(arg.type_diff, 'def')
	if exist('penalty_mex') == 3
		arg.type_diff = 'mex';
	else
		arg.type_diff = 'ind';
	end
end

% make sure mex can be supported if requested.  if not, revert gracefully
if streq(arg.type_diff, 'mex') && exist('penalty_mex') ~= 3
	warn 'no penalty_mex so "mex" option unavailable; revert to slow "ind"'
	arg.type_diff = 'ind';
end

% identity matrix is special case
if arg.offset == 0
	if arg.order ~= 0, warn 'only order=0 for identity', end

	switch arg.type_diff
	case 'spmat'
		n = arg.nn;
		ob = sparse(1:n, 1:n, ones(n,1), n, n);
	case 'sparse'
		ob = diag_sp(ones(arg.nn,1));
	otherwise
		ob = Fatrix(arg.dim, arg, 'caller', 'Cdiff1:ident', ...
		'forw', @Cdiff1_ident_dup, 'back', @Cdiff1_ident_dup, ...
		'power', @Cdiff1_power, 'abs', @Cdiff1_abs);
	end
return
end

if ~any(arg.order == [1 2])
	error 'only 1st and 2nd order done (for nonzero offsets)'
end

switch arg.type_diff
case 'convn'
	arg = Cdiff1_convn_setup(arg);
	ob = Fatrix(arg.dim, arg, 'caller', 'Cdiff1:convn', ...
		'forw', @Cdiff1_convn_forw, 'back', @Cdiff1_convn_back, ...
		'power', @Cdiff1_power, 'abs', @Cdiff1_abs);

case 'ind'
	ob = Fatrix(arg.dim, arg, 'caller', 'Cdiff1:ind', ...
		'forw', @Cdiff1_ind_forw, 'back', @Cdiff1_ind_back, ...
		'power', @Cdiff1_power, 'abs', @Cdiff1_abs);

case 'mex'
	arg.isize32 = int32(length(arg.isize));
	arg.offset = int32(arg.offset);
	ob = Fatrix(arg.dim, arg, 'caller', 'Cdiff1:mex', ...
		'forw', @Cdiff1_mex_forw, 'back', @Cdiff1_mex_back, ...
		'power', @Cdiff1_power, 'abs', @Cdiff1_abs);

case 'sparse'
	arg = Cdiff1_sp_setup(arg);
	ob = Fatrix(arg.dim, arg, 'caller', 'Cdiff1:sparse', ...
		'forw', @Cdiff1_sp_forw, 'back', @Cdiff1_sp_back, ...
		'power', @Cdiff1_power, 'abs', @Cdiff1_abs);

case 'spmat'
	arg = Cdiff1_sp_setup(arg);
	ob = arg.C.arg.G;

otherwise
	fail('bad type %s', arg.type_diff)
end


%
% Cdiff1_abs()
% for abs(C)
function ob = Cdiff1_abs(ob)
ob.arg.is_abs = true;


%
% Cdiff1_power()
% for C.^2
function ob = Cdiff1_power(ob, p)
ob.arg.Cpower = ob.arg.Cpower * p;


%
% Cdiff1_ident_dup()
% y = I * x
%
function y = Cdiff1_ident_dup(arg, x)
y = x;


%
% Cdiff1_mex_forw()
% y = C * x
%
function y = Cdiff1_mex_forw(arg, x)

[x ei] = embed_in(x, true(arg.isize), arg.nn); % [(N) *L]

if arg.is_abs
	diff_str = sprintf('diff%d,forwA', arg.order);
else
	diff_str = sprintf('diff%d,forw%d', arg.order, arg.Cpower);
end

if isreal(x)
	y = penalty_mex(diff_str, single(x), arg.offset, arg.isize32);
	y = double6(y);
else
	yr = penalty_mex(diff_str, single(real(x)), arg.offset, arg.isize32);
	yi = penalty_mex(diff_str, single(imag(x)), arg.offset, arg.isize32);
	y = double6(yr) + 1i * double6(yi);
end
y = reshapee(y, arg.isize, ei.diml); % [(N) 1 (L)] to [(N) (L)] due to mex

y = ei.shape(y); % [*N (L)] or [(N) (L)]


%
% Cdiff1_mex_back()
% x = C' * y
%
function x = Cdiff1_mex_back(arg, y)

if arg.is_abs
	diff_str = sprintf('diff%d,backA', arg.order);
else
	diff_str = sprintf('diff%d,back%d', arg.order, arg.Cpower);
end

[y eo] = embed_out(y, arg.isize); % [(N) *L]
y = reshapee(y, [arg.isize 1 eo.diml]); % [(N) (L)] to [(N) 1 (L)] for mex

if isreal(y)
	x = penalty_mex(diff_str, single(y), arg.offset, arg.isize32);
else
	xr = penalty_mex(diff_str, single(real(y)), arg.offset, arg.isize32);
	xi = penalty_mex(diff_str, single(imag(y)), arg.offset, arg.isize32);
	x = double6(xr) + 1i * double6(xi);
end
x = double6(x);

x = eo.shape(x, true(arg.isize), arg.nn); % [*N (L)] or [(N) (L)]


%
% Cdiff1_ind_forw()
% y = C * x
% in and out are both [(N) (L)]
%
function y = Cdiff1_ind_forw(arg, x)

flag_array = 0;
if size(x,1) ~= arg.nn
	diml = size(x); diml = diml((1+length(arg.isize)):end);
	x = reshapee(x, arg.nn, []); % [*N *L]
	flag_array = 1;
end

y = zeros(size(x));
if arg.order == 1
	off = arg.offset;
	ii = (1+max(off,0)):(arg.nn+min(off,0));
	coef = (-1) .^ arg.Cpower;
	if arg.is_abs
		coef = abs(coef);
	end

	% timings: it seems that the indexing slows it down a lot!
	% oh well, too bad for windoze users without the mex files
%	cpu etic
%	y = x + coef * x;
%	cpu etoc 'with no index'

%	cpu etic
%	y = x(1:end-1) + x(2:end);
%	cpu etoc 'with wired index'

%	cpu etic
%	y = x(ii,:) + x(ii - off,:);
%	cpu etoc 'with no coef'

%	cpu etic
	y(ii,:) = x(ii,:) + coef * x(ii-off,:);
%	cpu etoc 'with coef'

else % order = 2
	off = abs(arg.offset);
	ii = (1+off):(arg.nn-off);
	coef = [2 -1] .^ arg.Cpower;
	if arg.is_abs
		coef = abs(coef);
	end
	y(ii,:) = coef(1)*x(ii,:) + coef(2)*(x(ii-off,:) + x(ii+off,:));
end

if flag_array
	y = reshapee(y, arg.isize, diml); % [(N) (L)]
end


%
% Cdiff1_ind_back()
% x = C' * y
%
function x = Cdiff1_ind_back(arg, y)

flag_array = 0;
if size(y,1) ~= arg.nn
	diml = size(y); diml = diml((1+length(arg.isize)):end);
	y = reshapee(y, arg.nn, []); % [*N *L]
	flag_array = 1;
end

x = zeros(size(y));
if arg.order == 1
	off = arg.offset;
	ii = (1+max(off,0)):(arg.nn+min(off,0));
	coef = (-1) .^ arg.Cpower;
	if arg.is_abs
		coef = abs(coef);
	end
%	y(ii,:) = coef(1) * x(ii,:) + coef(2) * x(ii - arg.offset,:);
	x(ii,:) = y(ii,:);
	x(ii-off,:) = x(ii-off,:) + coef * y(ii,:);

else % order = 2
	off = abs(arg.offset);
	ii = (1+off):(arg.nn-off);
	coef = [2 -1] .^ arg.Cpower;
	if arg.is_abs
		coef = abs(coef);
	end
%	y(ii,:) = coef(1)*x(ii,:) + coef(2)*x(ii-off,:) + coef(2)*x(ii+off,:);
	x(ii,:) = coef(1) * y(ii,:);
	x(ii-off,:) = x(ii-off,:) + coef(2) * y(ii,:);
	x(ii+off,:) = x(ii+off,:) + coef(2) * y(ii,:);
end

if flag_array
	x = reshapee(x, arg.isize, diml); % [(N) (L)]
end


%
% Cdiff1_convn_setup()
% precompute filter coefficients for 'convn' version
%
function arg = Cdiff1_convn_setup(arg)

dd = penalty_displace(arg.offset, arg.isize);
if any(dd < 0), fail('need dd >= 0'), end
cdim = max(arg.order*dd+1, 1);
coef = zeros(cdim);

switch arg.order
case 0
	coef = 1;

case 1 % [1 -1]
	coef(1) = 1;
	coef(end) = -1;

case 2 % [-1 2 -1];
	coef(1) = -1;
	coef((end+1)/2) = 2;
	coef(end) = -1;

otherwise
	fail('order')
end

arg.coef = coef;


%
% Cdiff1_convn_forw()
% y = C * x
% x and y are both either [*N (L)] or [(N) (L)]
%
function y = Cdiff1_convn_forw(arg, x)

nd = length(arg.isize);

[x ei] = embed_in(x, true(arg.isize), arg.nn); % [(N) *L]
LL = prod(ei.diml);

coef = arg.coef;
coef = coef .^ arg.Cpower;
if arg.is_abs
	coef = abs(coef);
end

if LL == 1
	y = convn(x, coef, 'valid');
	y = padn(y, arg.isize);
else
	y = zeros(arg.nn, LL);
	for ll=1:LL
		tmp = stackpick(x,ll);
		tmp = convn(tmp, coef, 'valid');
		tmp = padn(tmp, arg.isize);
		y(:,ll) = tmp(:);
	end
	y = reshape(y, [arg.isize LL]);
end

y = ei.shape(y); % [*N (L)] or [(N) (L)]


%
% Cdiff1_convn_back()
% x = C' * y
% x and y are both either [*N (L)] or [(N) (L)]
%
function x = Cdiff1_convn_back(arg, y)

warn 'not working; it is too difficult to get the edge conditions to match'
arg.coef = flipdims(arg.coef); % trick
x = Cdiff1_convn_forw(arg, y);


%
% Cdiff1_sp_setup()
% setup for sparse version
%
function arg = Cdiff1_sp_setup(arg)

if arg.order == 1
	ii = (1+max(arg.offset,0)):(arg.nn+min(arg.offset,0));
	arg.C	= sparse(ii, ii, 1, arg.nn, arg.nn) ...
		- sparse(ii, ii-arg.offset, 1, arg.nn, arg.nn);

else % order = 2
	off = abs(arg.offset);
	ii = (1+off):(arg.nn-off);
	arg.C	= sparse(ii, ii, 2, arg.nn, arg.nn) ...
		- sparse(ii, ii-arg.offset, 1, arg.nn, arg.nn) ...
		- sparse(ii, ii+arg.offset, 1, arg.nn, arg.nn);
end
arg.C = Gsparse(arg.C, 'idim', arg.isize, 'odim', arg.isize);


%
% Cdiff1_sp_forw()
% y = C * x
% in and out are both [(N) (L)]
%
function y = Cdiff1_sp_forw(arg, x)

C = arg.C;
if arg.is_abs
	C = abs(C);
end
if arg.Cpower ~= 1
	C = C .^ arg.Cpower;
end

y = C * x;


%
% Cdiff1_sp_back()
% x = C' * y
%
function x = Cdiff1_sp_back(arg, y)

C = arg.C;
if arg.is_abs
	C = abs(C);
end
if arg.Cpower ~= 1
	C = C .^ arg.Cpower;
end

x = C' * y;
