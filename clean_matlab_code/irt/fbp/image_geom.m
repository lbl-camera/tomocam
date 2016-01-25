  function st = image_geom(varargin)
%|function st = image_geom(varargin)
%|
%| Create a "image geometry" structure that describes the sampling
%| characteristics of a single 2d image.
%| Using this structure should facilitate "object oriented" code.
%|
%| options for 2d
%|	'nx'		image dimension
%|	'ny'		image dimension (default: nx)
%|	'dx'		pixel size (required)
%|	'dy'		pixel size (default: -dx).  (can be a value or 'dx')
%|	'offset_x'	unitless (default: 0)
%|	'offset_y'	unitless (default: 0)
%|	'fov'		nx * dx
%|	'down'		down-sampling factor
%|	'mask'		logical support mask
%|	'offsets'	'dsp' for [-n/2:n/2-1] offsets, i.e., offset_* = 0.5
%|
%| options (additional) for 3d
%|	'nz'
%|	'dz'		(default: dx)
%|	'zfov'		dz * nz
%|	'offset_z'	unitless (default: 0)
%|
%| out:
%|	st	(strum)	initialized structure with methods
%|
%| methods:
%|	st.dim	[nx ny [nz]]
%|	st.x	1D x coordinates of each pixel (1D)
%|	st.y	1D y ""
%|	st.u,v	1D frequency-domain coordinates
%|	st.fg	{u v w} frequency-domain coordinates for DFT
%|	st.xg	x coordinates of each pixel, as a grid (2D or 3D)
%|	st.yg	y ""
%|	st.zg	z ""
%|	st.fovs	[|dx|*nx |dy|*ny ...] 
%|	st.np	sum(st.mask(:)) (# of pixels to be estimated)
%|	st.downsample(down_sampling_factor)
%|	st.embed(column)	turn short column(s) into array(s)
%|	st.maskit(x)		the opposite of embed
%|	st.shape(x)		reshape long column x to nx,ny array
%|	st.over(over_sampling_factor)
%|	st.unitv(ix,iy) | (jj) | ('c', [cx cy])
%|	st.ones | ('nz', nz) | (jj)
%|	st.zeros | ('nz', nz) | (jj)
%|	st.circ(rx,ry,cx,cy)	circle of radius rx,ry (cylinder for 3d)
%|
%| methods for 3d:
%|	st.z		z coordinates of each pixel
%|	st.mask_or	[nx,ny] logical "or" of mask
%|
%| Copyright 2006-1-19, Jeff Fessler, University of Michigan

if nargin == 1 && streq(varargin{1}, 'test'), image_geom_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end

% defaults
st.nx = [];
st.ny = [];
st.nz = [];
st.dx = [];
st.dy = [];
st.dz = [];
st.offset_x = 0;
st.offset_y = 0;
st.offset_z = 0;
st.offsets = '';
st.fov = [];
st.zfov = [];
st.down = 1;
st.mask = [];

st = vararg_pair(st, varargin);

% dimensions
if isempty(st.ny), st.ny = st.nx; end

if isempty(st.dz) && isempty(st.nz) && isempty(st.zfov)
%	st.type = '2d';
	st.is3 = false;
	st.dim = [st.nx st.ny];
	st = rmfield(st, {'nz', 'dz', 'zfov', 'offset_z'});
else
%	st.type = '3d';
	st.is3 = true;
	st.dim = [st.nx st.ny st.nz];
end

% offsets
if streq(st.offsets, 'dsp')
	if st.offset_x ~= 0 || st.offset_y ~= 0, fail('offsets usage'), end
	st.offset_x = 0.5;
	st.offset_y = 0.5;
	if st.is3
		if st.offset_z ~= 0 fail('offsets usage'), end
		st.offset_z = 0.5;
	end
end

% distances
if 1
	if isempty(st.dx)
		if isempty(st.fov), error 'dx or fov required', end
		if length(st.fov) == 1
			st.dx = st.fov / st.nx;
		elseif length(st.fov) == 2 && ~st.is3
			if ~isempty(st.dy), error 'dy and fov(2)?', end
			st.dx = st.fov(1) / st.nx;
			st.dy = st.fov(2) / st.ny;
			st.fov = st.fov(1);
		elseif length(st.fov) == 3 && st.is3
			if ~isempty(st.dy), error 'dy and fov(2)?', end
			if ~isempty(st.dz), error 'dz and fov(3)?', end
			if ~isempty(st.zfov), error 'zfov and fov(3)?', end
			st.dx = st.fov(1) / st.nx;
			st.dy = st.fov(2) / st.ny;
			st.dz = st.fov(3) / st.nz;
			st.zfov = st.fov(3);
			st.fov = st.fov(1);
		else
			fail('bad fov')
		end
	elseif isempty(st.fov) && ~isempty(st.nx)
		st.fov = st.nx * st.dx;
	end
	if st.fov ~= st.nx * st.dx
		error 'bad fov'
	end
end

if streq(st.dy, 'dx'), st.dy = st.dx; end % trick
if streq(st.dy, '-dx'), st.dy = -st.dx; end % trick
if isempty(st.dy), st.dy = -st.dx; end

if st.is3
	if isempty(st.nz)
		if ~isempty(st.dz) && ~isempty(st.zfov)
			st.nz = st.zfov / st.dz;
			if round(st.nz) ~= st.nz, error 'bad zfov/dz', end
			st.nz = round(st.nz);
		else
			error 'nz required for 3d'
		end
	end
	if isempty(st.dz)
		if isempty(st.zfov)
			st.dz = st.dx;
		elseif ~isempty(st.nz)
			st.dz = st.zfov / st.nz;
		else
			error 'need dz or zfov or nz'
		end
	end
	if isempty(st.zfov) && ~isempty(st.nz)
		st.zfov = st.nz * st.dz;
	end
	if st.zfov ~= st.nz * st.dz
		error 'bad zfov'
	end
end

% mask
if isempty(st.mask)
	st.mask = true(st.dim);
elseif ~islogical(st.mask)
	error 'mask must be logical'
elseif ndims(st.mask) ~= 2 + st.is3 ...
	|| size(st.mask,1) ~= st.nx ...
	|| size(st.mask,2) ~= st.ny ...
	|| (st.is3 && size(st.mask,3) ~= st.nz)
	size(st.mask), st.nx, st.ny
	error 'bad input mask size'
end

meth = { ...
	'circ', @image_geom_circ, '(rx, ry, cx, cy) [real units]'; ...
	'downsample', @image_geom_downsample, '()'; ...
	'embed', @image_geom_embed, '()'; ...
	'shape', @image_geom_shape, '(), out is [(N),L]'; ...
	'maskit', @image_geom_maskit, '(), opposite of embed'; ...
	'over', @image_geom_over, '()'; ...
	'x', @image_geom_x, '(subs), 1D x coordinates'; ...
	'y', @image_geom_y, '(subs), 1D y coordinates'; ...
	'fg', @image_geom_fg, '{subs}, cell of 2D or 3D frequency coordinates'; ...
	'xg', @image_geom_xg, '(subs), 2D or 3D grid of x coordinates'; ...
	'yg', @image_geom_yg, '(subs), 2D or 3D grid of y coordinates'; ...
	'zg', @image_geom_zg, '(subs), 2D or 3D grid of z coordinates'; ...
	'fovs', @image_geom_fovs, '(), field of views: [|dx|*nx |dy|*ny ...]'; ...
	'u', @image_geom_u, '(), 1D frequency domain coordinates'; ...
	'v', @image_geom_v, '()'; ...
	'unitv', @image_geom_unitv, ...
		'(ix,iy) | (ix,iy,iz) | (jj) | (''c'', [ox oy oz]) [ints]'; ...
	'zeros', @image_geom_zeros, '() | (''nz'', nz) | (jj)'; ...
	'ones', @image_geom_ones, '() | (''nz'', nz) | (jj)'; ...
	'np', @image_geom_np, '()'; ...
	};

if st.is3
	meth(end+[1:2],:) = { ...
		'z', @image_geom_z, '()'; ...
		'mask_or', @image_geom_mask_or, '()'};
end
st = strum(st, meth);

if st.down ~= 1
	down = st.down; st.down = 1; % trick
	st = st.downsample(down);
end


% image_geom_circ()
% default is a circle that just inscribes the square
% but keeping a 1 pixel border due to aspire regularization restriction
function circ = image_geom_circ(st, rad1, rad2, cx, cy)
if ~isvar('rad1') || isempty(rad1)
	rad1 = min(abs((st.nx/2-1)*st.dx), abs((st.ny/2-1)*st.dy));
end
if ~isvar('rad2') || isempty(rad2), rad2 = rad1; end
if ~isvar('cx') || isempty(cx), cx = 0; end
if ~isvar('cy') || isempty(cy), cy = 0; end
circ = ellipse_im(st, [cx cy rad1 rad2 0 1]);
if st.is3
	circ = repmat(circ, [1 1 st.nz]); % cylinder
end

% image_geom_downsample()
function st = image_geom_downsample(st, down)
st.nx = 2 * round(st.nx / down / 2);
st.ny = 2 * round(st.ny / down / 2);
st.dx = st.dx * down;
st.dy = st.dy * down;
st.down = st.down * down;
st.dim = [st.nx st.ny];

if st.is3
	st.nz = 2 * round(st.nz / down / 2);
	st.dz = st.dz * down;
	st.dim = [st.nx st.ny st.nz];
end

mdim = size(st.mask);
if st.is3
	if all(st.dim * down == mdim)
		st.mask = downsample3(st.mask, down) > 0; % inclusive or
	elseif st.nx ~= size(st.mask,1) || st.ny ~= size(st.mask,2) ...
		|| st.nz ~= size(st.mask,3)
		error 'bug: bad mask size.  need to address mask downsampling'
	end
else
	if st.nx * down == size(st.mask,1) && st.ny * down == size(st.mask,2)
		st.mask = downsample2(st.mask, down) > 0; % inclusive or
	elseif st.nx ~= size(st.mask,1) || st.ny ~= size(st.mask,2)
		error 'bug: bad mask size.  need to address mask downsampling'
	end
end

% image_geom_mask_or()
function mo = image_geom_mask_or(st)
mo = sum(st.mask, 3) > 0;

% image_geom_np()
function np = image_geom_np(st)
np = sum(st.mask(:));


% image_geom_over()
function st = image_geom_over(st, over)
st.nx = st.nx * over;
st.ny = st.ny * over;
st.dx = st.dx / over;
st.dy = st.dy / over;
st.offset_x = st.offset_x * over;
st.offset_y = st.offset_y * over;
if st.is3
	st.nz = st.nz * over;
	st.dz = st.dz / over;
	st.offset_z = st.offset_z * over;
end

% image_geom_embed()
function x = image_geom_embed(st, x)
if issparse(x), x = image_geom_embed_sparse(st, x); return, end
x = embed(x, st.mask);

% image_geom_embed_sparse()
% trick: this is for "unpacking" sparse system matrices
function x = image_geom_embed_sparse(st, x)
[i j a] = find(x);
ind = find(st.mask);
j = ind(j);
x = sparse(i, j, a, size(x,1), prod(st.dim));

% image_geom_maskit()
% opposite of embed
% in 3d case, if input is [nx ny nz n4], output is [np n4] where np = sum(mask)
function x = image_geom_maskit(st, x)
dim = size(x);
x = reshape(x, prod(st.dim), []);
x = x(st.mask(:),:);
if length(dim) > length(st.dim)
	x = reshape(x, [], dim((1+length(st.dim)):end));
end

% image_geom_shape()
function x = image_geom_shape(st, x)
if st.is3
	x = reshape(x, st.nx, st.ny, st.nz, []);
else
	x = reshape(x, st.nx, st.ny, []);
end

% image_geom_x()
function x = image_geom_x(st, varargin)
wx = (st.nx-1)/2 + st.offset_x;
x = ([0:st.nx-1]' - wx) * st.dx;
x = x(varargin{:});

% image_geom_y()
function y = image_geom_y(st, varargin)
wy = (st.ny-1)/2 + st.offset_y;
y = ([0:st.ny-1]' - wy) * st.dy;
y = y(varargin{:});

% image_geom_z()
function z = image_geom_z(st, varargin)
wz = (st.nz-1)/2 + st.offset_z;
z = ([0:st.nz-1]' - wz) * st.dz;
z = z(varargin{:});


% DFT frequency sample grid
% image_geom_fg()
function fg = image_geom_fg(st, varargin);
u = image_geom_u(st);
v = image_geom_v(st);
if st.is3
	n = st.nz; d = st.dz;
	w = [-n/2:n/2-1] / (n * d);
	[u v w] = ndgrid(u, v, w);
	fg = {u, v, w};
else
	[u v] = ndgrid(u, v);
	fg = {u, v};
end
if length(varargin),
	fg = fg{varargin{:}};
end


% image_geom_xg()
function x = image_geom_xg(st, varargin);
x = image_geom_x(st);
x = x(:);
if st.is3
	x = repmat(x, [1 st.ny st.nz]);
else
	x = repmat(x, [1 st.ny]);
end
x = x(varargin{:});

% image_geom_yg()
function y = image_geom_yg(st, varargin);
y = image_geom_y(st);
y = y(:)';
if st.is3
	y = repmat(y, [st.nx 1 st.nz]);
else
	y = repmat(y, [st.nx 1]);
end
y = y(varargin{:});

% image_geom_zg()
function z = image_geom_zg(st, varargin);
if ~st.is3
	z = zeros(st.nx, st.ny);
	return
end
z = image_geom_z(st);
z = reshape(z, [1 1 st.nz]);
z = repmat(z, [st.nx st.ny 1]);
z = z(varargin{:});

% image_geom_fovs()
function out = image_geom_fovs(st, varargin)
out = [abs(st.dx) * st.nx abs(st.dy) * st.ny];
if st.is3
	out = [out abs(st.dz) * st.nz];
end
out = out(varargin{:});

% image_geom_u()
function out = image_geom_u(st, varargin)
n = st.nx; d = st.dx;
out = [-n/2:n/2-1] / (n * d);
out = out(varargin{:});

% image_geom_v()
function out = image_geom_v(st, varargin)
n = st.ny; d = st.dy;
out = [-n/2:n/2-1] / (n * d);
out = out(varargin{:});

% image_geom_zeros()
function z = image_geom_zeros(st, varargin)
if length(varargin) && ischar(varargin{1})
	arg.nz = 1;
	arg = vararg_pair(arg, varargin);
	z = zeros([st.dim arg.nz], 'single');
else
	z = zeros(st.dim, 'single');
	z = z(varargin{:});
end

% image_geom_ones()
function o = image_geom_ones(st, varargin)
if length(varargin) && ischar(varargin{1})
	arg.nz = 1;
	arg = vararg_pair(arg, varargin);
	o = ones([st.dim arg.nz], 'single');
else
	o = ones(st.dim, 'single');
	o = o(varargin{:});
end

% image_geom_unitv()
% use: st.unitv(ix,iy) or st.unitv(ix,iy,iz) or st.unitv(jj)
% or st.unitv('c', [ix_offset iy_offset iz_offset])
% plain st.unitv defaults to "center" of image
function ej = image_geom_unitv(st, varargin)
ej = zeros(st.dim, 'single');
if ~length(varargin)
	t = num2cell(floor(st.dim/2+1)); % "center"
	ej(t{:}) = 1;
elseif streq(varargin{1}, ':')
	t = num2cell(floor(st.dim/2+1)); % "center"
	ej(t{:}) = 1;
	ej = ej(:);
elseif streq(varargin{1}, 'c')
	t = floor(st.dim/2+1) + varargin{2}; % "offset"
	t = num2cell(t);
	ej(t{:}) = 1;
else
	ej(varargin{:}) = 1;
end


%
% image_geom_test()
%
function image_geom_test
%st = image_geom('nx', 8, 'fov', [2 4])
st = image_geom('nx', 8, 'nz', 4, 'dx', 2, 'mask', true(8,8,4));
st.nx;
st.np;
st.x';
st.u;
st.v;
st.fg;
st.xg + st.yg + st.zg;
st.y';
st.mask;
st.circ;
st.mask(3,4:6,1);
size(st.maskit(repmat(st.ones, [1 1 1 3])));
st.over(2);
st.zeros;
st.unitv(1,1);
st.mask_or;
st.downsample(2);
st.fovs;

%ig = image_geom('nx', 8, 'dx', 2);
%im(ig.x, ig.y, ig.circ)
