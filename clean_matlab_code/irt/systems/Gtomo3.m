  function ob = Gtomo3(sys_type, mask, nx, ny, nz, varargin)
%|function ob = Gtomo3(sys_type, mask, nx, ny, nz, options)
%|
%| alternate usage for CT:
%|function ob = Gtomo3(cg, ig, options)
%|
%| Construct Gtomo3 object, for 3D tomographic image reconstruction.
%| This object does 3d forward and backprojection using ASPIRE.
%| See Gtomo3_test.m for example usage.
%|
%| Basically, you create an object calling:
%|		G = Gtomo3(sys_type, mask)
%| and then you can use it thereafter by typing commands like
%|		y = G * x;
%| which will auto-magically do the multiplication by calling a mex file.
%|
%| in
%|	sys_type	string	see 3D ASPIRE User's Guide
%|	mask	[nx,ny,nz] or string
%|		mask can be a filename, in which case the file is read.
%|		mask can be empty (for some system types, including 3s),
%|		in which case it is determined from backprojecting a "1" vector.
%|		Else, mask must be [nx,ny,nz] binary array of object support.
%| options
%|	'nthread' 1, 2...	1, unless you have a multi-processor system
%|	'chat' 0|1		verbosity.  default: 1.
%|	'checkmask' 0|1		verify that the mask is correct.  default: 1.
%|	'view2d' 0|1		for 2z and 2dsc system types, use 2d views
%|				instead of sinograms, for use with OS.
%|				default: 0
%|	'permute213' 0|1	convert from 123 to 213 output order
%|				(e.g., to overcome vu vs st order of 3l@)
%|				default: 0
%|	'scale' 1	 	scale factor.  default 1
%|
%| out
%|	ob [nd,np]	np = sum(mask(:)), so it is already "masked"
%|
%| For more help, type 'f3d_mex' and see Gtomo3_test.m
%|
%| Copyright 01-04-22, Jeff Fessler, University of Michigan

if nargin == 1 & streq(sys_type, 'test'), Gtomo3_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end

% handle new usage (for CT)
% (cg, ig, [options])
if isa(sys_type, 'strum')
	cg = sys_type;
	ig = mask;
	options = {};
	if isvar('nx')
		options = {nx};
		if isvar('ny')
			options = {nx, ny};
			if isvar('nz')
				options = {nx, ny, nz};
			end
		end
	end
	options = {options{:}, varargin{:}};

	sys_type = aspire_pair(cg, ig, 'system', '3l');
	ob = Gtomo3(sys_type, ig.mask, ig.nx, ig.ny, ig.nz, options{:});
return
end

arg.sys_type = sys_type;
arg.nx = nx;
arg.ny = ny;
arg.nz = nz;
arg.nxyz = nx*ny*nz;
arg.mask = logical(mask);

% defaults
arg.nthread = 1;
arg.chat = true;
arg.checkmask = true;
arg.scale = 1;
arg.view2d = false;
arg.permute213 = false;

if ischar(arg.mask)
	arg.mask = read(arg.mask); % read mask file
end

% options
arg = vararg_pair(arg, varargin);

if arg.view2d & ~streq(arg.sys_type, '2', 1)
	error 'view2d option only relevant to 2dsc and 2z system types'
end
if arg.permute213 & ~streq(arg.sys_type, '3l', 2)
	warn 'permute213 designed only for "3l" system model!?'
end

%
% if no mask, then backproject to find it!
%
if isempty(arg.mask)
	f3d_mex('init', arg.sys_type, uint8(ones(nx,ny,nz)), ...
		int32(arg.nx), int32(arg.ny), int32(arg.nz), ...
		int32(arg.nthread), int32(arg.chat))
	warning('creating mask by summing will be a bit slow')
	t = double(f3d_mex('dims')); % data dimensions, matlab wants doubles
	arg.mask = f3d_mex('back', single(ones(t)), ...
			int32(0), int32(1), int32(0)) > 0;
	f3d_mex('free')
end

% initialize mex file
f3d_mex('init', arg.sys_type, uint8(arg.mask), ...
	int32(arg.nx), int32(arg.ny), int32(arg.nz), ...
	int32(arg.nthread), int32(arg.chat))

arg.nn = Gtomo3_nn(arg);
arg.nd = prod(arg.nn);
arg.np = sum(arg.mask(:));
dim = [arg.nd arg.np]; % trick: make it masked by default!

%
% build Fatrix object
%
if arg.scale == 1
	after = {};
else
	after = {'cascade_after', arg.scale};
end

% trick: handle 3l@ uv vs st mismatch (if requested) by permuting.
if arg.permute213
%	if ~isempty(after), fail('permute213 and scale not done'), end
%	after = @(x, is_transpose, istart, nblock) permute(x, [2 1 3]);
%	after = {'cascade_after', after};
	warn 'still testing'
end

ob = Fatrix(dim, arg, 'caller', 'Gtomo3', ...
	'forw', @Gtomo3_forw, 'back', @Gtomo3_back, ...
	'free', @Gtomo3_free, 'mtimes_block', @Gtomo3_mtimes_block, ...
	after{:});

if arg.chat
	f3d_mex('show', int32(arg.chat));	% display header info
end

% check mask
if arg.checkmask
	printm('Warn %s.m: checking mask, which may be unnecessary', mfilename)
	tmp = sum(ob).' > 0;
	tmp = embed(tmp, arg.mask);
%	tmp = reshape(tmp > 0, [arg.nx arg.ny arg.nz]);
%	if ndims(tmp) ~= ndims(arg.mask), error 'wrong mask dims', end
%	if any(size(tmp) ~= size(arg.mask)), error 'wrong mask size', end
	if any(tmp(:) ~= arg.mask(:))
		tmpfill = imfill(tmp);
		if any(tmpfill(:) ~= arg.mask(:))
			if im
				figure(1), clf, im(tmp, 'computed mask')
				figure(2), clf, im(arg.mask, 'input mask')
			end
			printm('Warn: mask-sum mismatch not corrected by fill')
		else
			printm('Warn: mask-sum mismatch; corrected by filling')
		end
		printm('Warn: probably means this geometry has poor sampling')
%		printm('range(tmp,2)'), disp(range(arg.mask,2)')
%		disp('examine mask images then hit any key to continue:')
%		pause
	end
	printm('mask checking completed')
end


%
% Gtomo3_nn(): data dimensions
%
function nn = Gtomo3_nn(arg)
nn = double(f3d_mex('dims')); % silly matlab wants size to be double
if arg.view2d
	nn = nn([1 3 2]); % swap projection angle and slice index
end
if arg.permute213
	nn = nn([2 1 3]); % trick: swap vu to st for 3l@
end


%
% Gtomo3_forw(): y = G * x
% full projection
%
function y = Gtomo3_forw(arg, x)
y = Gtomo3_mtimes_block(arg, 0, x, 1, 1);


%
% Gtomo3_back(): x = G' * y
% full backprojection
%
function x = Gtomo3_back(arg, y)
x = Gtomo3_mtimes_block(arg, 1, y, 1, 1);


%
% Gtomo3_mtimes_block()
%
function y = Gtomo3_mtimes_block(arg, is_transpose, x, istart, nblock)
if is_transpose
	y = Gtomo3_block_back(arg, x, istart, nblock);
else
	y = Gtomo3_block_forw(arg, x, istart, nblock);
end


%
% Gtomo3_block_forw()
%
function y = Gtomo3_block_forw(arg, x, istart, nblock)

[x ei] = embed_in(x, arg.mask, arg.np);

for i4=1:size(x,4)
	y(:,:,:,i4) = f3d_mex('proj', single(x(:,:,:,i4)), ...
		int32(istart-1), int32(nblock), int32(arg.chat));
end
y = double6(y);
if arg.view2d % trick: for 2z and 2dsc
	y = permute(y, [1 3 2 4]);
end
if arg.permute213 % trick: for 3l@
	y = permute(y, [2 1 3 4]);
end
na = arg.nn(end); % last index must be the one over which we subsetize
ia = istart:nblock:na;
y = reshape(y, prod(arg.nn(1:end-1)), na, []);
LL = size(y,3);
y = y(:,ia,:);
y = reshape(y, [arg.nn(1:end-1) length(ia) LL]); % [(M) *L]
y = ei.shape(y);


%
% Gtomo3_block_back()
%
function x = Gtomo3_block_back(arg, y, istart, nblock)

na = arg.nn(end); % last index must be the one over which we subsetize
ia = istart:nblock:na;

[y eo] = embed_out(y, [arg.nn(1:end-1) length(ia)]); % [(M) *L]

n1 = prod(arg.nn(1:end-1));
y = reshape(y, n1, length(ia), []);
yy = size(y); yy(2) = na; yy = zeros(yy);
yy(:,ia,:) = y;
dim = num2cell(arg.nn);
y = reshape(yy, dim{:}, []);

if arg.permute213 % trick: for 3l@
	y = permute(y, [2 1 3 4]);
end
if arg.view2d % trick for 2d
	y = permute(y, [1 3 2 4]);
end
for i4 = 1:size(y,4)
	x(:,:,:,i4) = f3d_mex('back', single(y(:,:,:,i4)), ...
		int32(istart-1), int32(nblock), int32(arg.chat));
end
x = double6(x);
x = eo.shape(x, arg.mask, arg.np);


%
% Gtomo3_free()
%
function Gtomo3_free(arg)
if arg.chat
	disp('freeing Gtomo3 object static memory')
end
f3d_mex('free');
