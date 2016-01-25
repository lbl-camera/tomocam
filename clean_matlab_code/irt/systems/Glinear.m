 function G = Glinear(varargin)
%function G = Glinear(options)
%
% Generate a geometric system matrix for tomographic projection
% based on simple linear interpolation.
% Contains exactly two g_{ij}'s per pixel per projection angle.
% This system model is pretty inadequate for reconstructing
% real tomographic data, but is useful for simple simulations.
%
% options:
%	nx,ny		image size
%	nb,na		sinogram size (n_radial_bins X n_angles).
%	ray_pix		ray_spacing / pixel_size (usually 1 or 0.5)
%	mask [nx,ny]	which pixels are to be reconstructed
%	chat		verbosity
% out:
%	G [nb*na,nx*ny]	all of G, regardless of mask
%			since entire size is needed for .wtf saving.
%
% Caller must do G = G(:,mask(:)) for masked reconstructions.
%
% Copyright Apr 2000, Jeff Fessler, University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if nargin == 1 && streq(varargin{1}, 'test'), Glinear_test, return, end

% defaults
arg.nx = 32;
arg.ny = [];
arg.nb = [];
arg.na = [];
arg.ray_pix = 1;
arg.mask = [];
arg.chat = false;

% old-style arguments: nx, ny, nb, na, ray_pix, mask, chat
if isnumeric(varargin{1})
	narg = length(varargin);
	if narg >= 1, arg.nx = varargin{1}; end
	if narg >= 2, arg.ny = varargin{2}; end
	if narg >= 3, arg.nb = varargin{3}; end
	if narg >= 4, arg.na = varargin{4}; end
	if narg >= 5, arg.ray_pix = varargin{5}; end
	if narg >= 6, arg.mask = varargin{6}; end
	if narg >= 7, arg.chat = varargin{7}; end
else
	arg = vararg_pair(arg, varargin);
end

if isempty(arg.ny), arg.ny = arg.nx; end
if isempty(arg.nb), arg.nb = arg.nx; end
if isempty(arg.na), arg.na = floor(arg.nb * pi/2); end
if isempty(arg.mask), arg.mask = true(arg.nx,arg.ny); end

G = Glinear_do(arg.nx, arg.ny, arg.nb, arg.na, arg.ray_pix, arg.mask, arg.chat);

function G = Glinear_do(nx, ny, nb, na, ray_pix, mask, chat);

if ray_pix < 0.5
	warning('small ray_pix will give lousy results!')
end

%
% pixel centers
%
x = [0:nx-1] - (nx-1)/2;
y = (-1)*([0:ny-1] - (ny-1)/2);		% trick: to match aspire
[x,y] = ndgrid(x, y);
x = x(mask(:));
y = y(mask(:));
np = length(x);		% sum(mask(:)) - total # of support pixels

angle = [0:na-1]'/na * pi;
tau = cos(angle) * x' + sin(angle) * y';	% [na,np] projected pixel center
tau = tau / ray_pix;		% account for ray_spacing / pixel_size
tau = tau + (nb+1)/2;		% counting from 1 (matlab)
ibl = floor(tau);		% left bin
val = 1 - (tau-ibl);		% weight value for left bin

ii = ibl + [0:na-1]'*nb*ones(1,np);	% left sinogram index

good = ibl(:) >= 1 & ibl(:) < nb;	% within FOV cases
if any(~good), warning 'FOV too small', end

%np = sum(mask(:));
%nc = np;	jj = 1:np;		% compact G
nc = nx * ny;	jj = find(mask(:))';	% all-column G
jj = jj(ones(1,na),:);

val1 = 1-val;
if 0	% make precision match aspire?
	val = double(single(val));
	val1 = double(single(val1));
end

G1 = sparse(ii(good), jj(good), val(good), nb*na, nc);		% left bin
G2 = sparse(ii(good)+1, jj(good), val1(good), nb*na, nc);	% right bin
G = G1 + G2;

if 0
%	subplot(121), im(embed(sum(G)', mask))		% for compact
	subplot(121), im(reshape(sum(G), nx, ny))	% for all-column
	subplot(122), im(reshape(sum(G'), nb, na))
end

%
% test demo
%
function Glinear_test
nx = 32; ny = nx; nb = 40; na = 42;
x = shepplogan(nx, ny, 1);
ix = [-(nx-1)/2:(nx-1)/2]';
iy = [-(ny-1)/2:(ny-1)/2];
rr = sqrt(outer_sum(ix.^2, iy.^2));
mask = rr < nx/2-1;
im clf, im pl 2 2
%G = Glinear(nx, ny, nb, na, 1., mask); % old syntax
G = Glinear('nx', nx, 'ny', ny, 'nb', nb, 'na', na, 'mask', mask);
y = G * x(:);		% forward projection
y = reshape(y, nb, na);	% reshape into sinogram array
sino = zeros(nb, na);	sino(nb/2, 10) = 1;
b = reshape(G' * sino(:), nx, ny);
im(1, mask, 'support mask')
im(2, x, 'test image')
im(3, y, 'sinogram'), xlabel ib, ylabel ia
im(4, b, 'backproject 1 ray')
