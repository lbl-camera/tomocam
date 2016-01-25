  function [phantom, params] = ellipsoid_im(ig, params, varargin)
%|function [phantom, params] = ellipsoid_im(ig, params, varargin)
%|
%| generate ellipsoids phantom image from parameters:
%|	[x_center y_center z_center x_radius y_radius z_radius
%|		xy_angle_degrees z_angle_degrees amplitude]
%| in
%|	ig		image_geom()
%|	params [ne,9]	ellipsoid parameters.  if empty, use 3d shepp-logan
%|			[x_center y_center z_center  x_radius y_radius z_radius
%|				xy_angle_degrees z_angle_degrees  amplitude]
%| option
%|	'oversample'	over-sampling factor (default: 1)
%|	'checkfov'	0|1 warn if any ellipsoid is out of fov
%|	'fast'	0|1	1 means use new 'fast' (?) version (default: 0)
%| out
%|	phantom		[nx,ny,nz] image
%|
%| op ellipsoid in aspire with nint=3 is oversample=4 = 2^(3-1) here
%|
%| Copyright 2004-8-13, Patty Laskowsky, Nicole Caparanis, Taka Masuda,
%| and Jeff Fessler, University of Michigan

if nargin == 1 && streq(ig, 'test'), ellipsoid_im_test, return, end
if nargin == 1 && streq(ig, 'profile'), ellipsoid_im_profile, return, end
if nargin < 2, help(mfilename), error(mfilename), end

if isnumeric(ig)
	[phantom params] = ellipsoid_im_old(ig, params, varargin{:});
return
end

arg.oversample = 1;
arg.checkfov = false;
arg.fast = false;
arg = vararg_pair(arg, varargin);

if ~isvar('params') || isempty(params)
	params = 'shepp-logan';
end

if ischar(params)
	params = shepp_logan_3d_parameters(ig.fov/2, ig.fov/2, ig.zfov/2, params);
end

if arg.fast && arg.oversample > 1
	fun = @ellipsoid_im_fast;
else
	fun = @ellipsoid_im_slow;
end

[phantom params] = fun(ig.nx, ig.ny, ig.nz, params, ...
	ig.dx, ig.dy, ig.dz, ig.offset_x, ig.offset_y, ig.offset_z, ...
	arg.oversample, arg.checkfov);
end % ellipsoid_im


%
% ellipsoid_im_old()
%
function [phantom, params] = ellipsoid_im_old(nx, ny, nz, params, ...
	dx, dy, dz, varargin)

if nargout > 1, warning 'units of output params not finished', end

arg.oversample = 1;
arg = vararg_pair(arg, varargin);

if ~isvar('ny') || isempty(ny), ny = nx; end
if ~isvar('nz') || isempty(nz), nz = nx; end

if ~isvar('dx') || isempty(dx), dx = 1; end
if ~isvar('dy') || isempty(dy), dy = dx; end
if ~isvar('dz') || isempty(dz), dz = dx; end

if ~isvar('params') || isempty(params), params = 'shepp-logan'; end

[phantom params] = ellipsoid_im_slow(nx, ny, nz, params, ...
	dx, dy, dz, 0, 0, 0, arg.oversample, false);
end % ellipsoid_im_old()


%
% ellipsoid_im_slow()
%
function [phantom params] = ellipsoid_im_slow(nx, ny, nz, params, ...
	dx, dy, dz, offset_x, offset_y, offset_z, over, checkfov)

if size(params,2) ~= 9
	error 'bad ellipse parameter vector size'
end

phantom = zeros(nx*over, ny*over, nz*over, 'single');

wx = (nx*over-1)/2 + offset_x*over;
wy = (ny*over-1)/2 + offset_y*over;
wz = (nz*over-1)/2 + offset_z*over;
xx = ((0:nx*over-1) - wx) * dx / over;
yy = ((0:ny*over-1) - wy) * dy / over;
zz = ((0:nz*over-1) - wz) * dz / over;
xmax = max(xx); xmin = min(xx);
ymax = max(yy); ymin = min(yy);
zmax = max(zz); zmin = min(zz);
[xx yy zz] = ndgrid(xx, yy, zz);

ticker reset
ne = nrow(params);
for ie = 1:ne;
	ticker(mfilename, ie, ne)

	ell = params(ie, :);
	cx = ell(1);	rx = ell(4);
	cy = ell(2);	ry = ell(5);
	cz = ell(3);	rz = ell(6);

	theta = deg2rad(ell(7));
	phi = deg2rad(ell(8));
	[xr yr zr] = rot3(xx-cx, yy-cy, zz-cz, theta, phi);

	tmp = (xr / rx).^2 + (yr / ry).^2 + (zr / rz).^2 <= 1;
	phantom = phantom + ell(9) * tmp;

	if checkfov
		if cx + rx > xmax || cx - rx < xmin
			warn('fov: x range %g %g, cx=%g rx=%g', xmin, xmax, cx, rx)
		end
		if cy + ry > ymax || cy - ry < ymin
			warn('fov: y range %g %g, cy=%g ry=%g', ymin, ymax, cy, ry)
		end
		if cz + rz > zmax || cz - rz < zmin
			warn('fov: z range %g %g, cz=%g rz=%g', zmin, zmax, cz, rz)
		end
	end
end

phantom = downsample3(phantom, over);
end % ellipsoid_im_slow()


%
% ellipsoid_im_fast()
%
function [phantom params] = ellipsoid_im_fast(nx, ny, nz, params, ...
	dx, dy, dz, offset_x, offset_y, offset_z, over, checkfov)

if size(params,2) ~= 9
	error 'bad ellipse parameter vector size'
end

phantom = zeros(nx, ny, nz, 'single');

wx = (nx-1)/2 + offset_x;
wy = (ny-1)/2 + offset_y;
wz = (nz-1)/2 + offset_z;
xx = ((0:nx-1) - wx) * dx;
yy = ((0:ny-1) - wy) * dy;
zz = ((0:nz-1) - wz) * dz;
xmax = max(xx); xmin = min(xx);
ymax = max(yy); ymin = min(yy);
zmax = max(zz); zmin = min(zz);
[xx yy zz] = ndgrid(xx, yy, zz);

ne = nrow(params);
if checkfov
	for ie = 1:ne
		ell = params(ie, :);
		cx = ell(1);	rx = ell(4);
		cy = ell(2);	ry = ell(5);
		cz = ell(3);	rz = ell(6);

		if cx + rx > xmax || cx - rx < xmin
			warn('fov: x range %g %g, cx=%g rx=%g', xmin, xmax, cx, rx)
		end
		if cy + ry > ymax || cy - ry < ymin
			warn('fov: y range %g %g, cy=%g ry=%g', ymin, ymax, cy, ry)
		end
		if cz + rz > zmax || cz - rz < zmin
			warn('fov: z range %g %g, cz=%g rz=%g', zmin, zmax, cz, rz)
		end
	end
end

if over > 1
	tmp = ((1:over) - (over+1)/2) / over;
	[xf yf zf] = ndgrid(tmp*dx, tmp*dy, tmp*dz);
	xf = xf(:)';
	yf = yf(:)';
	zf = zf(:)';

	hx = abs(dx) / 2;
	hy = abs(dy) / 2;
	hz = abs(dz) / 2;
end

ticker reset
for ie = 1:ne;
	ticker(mfilename, ie, ne)

	ell = params(ie, :);
	cx = ell(1);	rx = ell(4);
	cy = ell(2);	ry = ell(5);
	cz = ell(3);	rz = ell(6);

	theta = deg2rad(ell(7));
	phi = deg2rad(ell(8));

	xs = xx - cx; % shift per center
	ys = yy - cy;
	zs = zz - cz;

	% coordinates of "outer" corner of each voxel, relative to ellipsoid center
        xo = xs + sign(xs) * hx;
	yo = ys + sign(ys) * hy;
	zo = zs + sign(zs) * hz;

	% voxels that are entirely inside the ellipse:
	[xr yr zr] = rot3(xo, yo, zo, theta, phi);
	vi = (xr / rx).^2 + (yr / ry).^2 + (zr / rz).^2 <= 1;
	gray = single(vi);

	if over > 1
		% coordinates of "inner" corner of each pixel, relative to ellipse center
		xi = xs - sign(xs) * hx;
		yi = ys - sign(ys) * hy;
		zi = zs - sign(zs) * hz;

		% voxels that are entirely outside the ellipsoid:
		[xr yr zr] = rot3(xi, yi, zi, theta, phi);
		vo = (max(abs(xr),0) / rx).^2 + (max(abs(yr),0) / ry).^2 ...
			+ (max(abs(zr),0) / rz).^2 >= 1;

		% subsampling for edge voxels
		edge = ~vi & ~vo;
		x = xx(edge);
		y = yy(edge);
		z = zz(edge);
		x = outer_sum(x, xf);
		y = outer_sum(y, yf);
		z = outer_sum(z, zf);

		[xr yr zr] = rot3(x - cx, y - cy, z - cz, theta, phi);
		in = (xr / rx).^2 + (yr / ry).^2 + (zr / rz).^2 <= 1;
		tmp = mean(in, 2);

		gray(edge) = tmp;
	end

	phantom = phantom + ell(9) * gray;

end

end % ellipsoid_im_fast()


%
% rot3()
%
function [xr, yr, zr] = rot3(x, y, z, theta, phi)
if phi, error 'z rotation not done', end
xr =  cos(theta) * x + sin(theta) * y;
yr = -sin(theta) * x + cos(theta) * y;
zr = z;
end % rot3()


%
% shepp_logan_3d_parameters()
% most of these values are unitless "fractions of field of view"
%
function params = shepp_logan_3d_parameters(xfov, yfov, zfov, ptype)

% parameters from Kak and Slaney text, p. 102, which seem to have typos!
ekak = [...
	0	0	0	0.69	0.92	0.9	0	2.0;
	0	0	0	0.6624	0.874	0.88	0	-0.98;
	-0.22	0	-0.25	0.41	0.16	0.21	108	-0.02;
	0.22	0	-0.25	0.31	0.11	0.22	72	-0.02;
	0	0.1	-0.25	0.046	0.046	0.046	0	0.02; % same?
	0	0.1	-0.25	0.046	0.046	0.046	0	0.02; % same?
	-0.8	-0.65	-0.25	0.046	0.023	0.02	0	0.01;
	0.06	-0.065	-0.25	0.046	0.023	0.02	90	0.01;
	0.06	-0.105	0.625	0.56	0.04	0.1	90	0.02;
	0	0.1	-0.625	0.056	0.056	0.1	0	-0.02];

% the following parameters came from leizhu@stanford.edu
% who says that the Kak&Slaney values are incorrect
% fix: i haven't had time to look into this in detail
% yu:05:ads cites shepp:74:tfr 

%	x	y	z	rx	ry	rz	angle	density
ezhu = [...
	0	0	0	0.69	0.92	0.9	0	2.0;
	0	-0.0184	0	0.6624	0.874	0.88	0	-0.98;
	-0.22	0	-0.25	0.41	0.16	0.21	-72	-0.02;
	0.22	0	-0.25	0.31	0.11	0.22	72	-0.02;
	0	0.35	-0.25	0.21	0.25	0.35	0	0.01;
	0	0.1	-0.25	0.046	0.046	0.046	0	0.01;
	-0.08	-0.605	-0.25	0.046	0.023	0.02	0	0.01;
	0	-0.1	-0.25	0.046	0.046	0.046	0	0.01;
	0	-0.605	-0.25	0.023	0.023	0.023	0	0.01;
	0.06	-0.605	-0.25	0.046	0.023	0.02	-90	0.01;
	0.06	-0.105	0.0625	0.056	0.04	0.1	-90	0.02;
	0	0.1	0.625	0.056	0.056	0.1	0	-0.02];

% and here are parameters from the "phantom3d.m" in matlab central
% by Matthias Schabel matlab@schabel-family.org
% which cites p199-200 of peter toft thesis: http://petertoft.dk/PhD/
% but that thesis has only 2d phantom!
%
% e(:,1) = [1 -.98 -.02 -.02 .01 .01 .01 .01 .01 .01];
%
%     Column 1:  A      the additive intensity value of the ellipsoid
%     Column 2:  a      the length of the x semi-axis of the ellipsoid 
%     Column 3:  b      the length of the y semi-axis of the ellipsoid
%     Column 4:  c      the length of the z semi-axis of the ellipsoid
%     Column 5:  x0     the x-coordinate of the center of the ellipsoid
%     Column 6:  y0     the y-coordinate of the center of the ellipsoid
%     Column 7:  z0     the z-coordinate of the center of the ellipsoid
%     Column 8:  phi    phi Euler angle (in degrees) (rotation about z-axis)
%     Column 9:  theta  theta Euler angle (in degrees) (rotation about x-axis)
%     Column 10: psi    psi Euler angle (in degrees) (rotation about z-axis)
%
%   For purposes of generating the phantom, the domains for the x-, y-, and 
%   z-axes span [-1,1].  Columns 2 through 7 must be specified in terms
%   of this range.
%
%         A     a    b    c     x0      y0      z0    phi  theta    psi
%        -----------------------------------------------------------------
e3d =  [  1 .6900 .920 .810      0       0       0      0      0      0
        -.8 .6624 .874 .780      0  -.0184       0      0      0      0
        -.2 .1100 .310 .220    .22       0       0    -18      0     10
        -.2 .1600 .410 .280   -.22       0       0     18      0     10
         .1 .2100 .250 .410      0     .35    -.15      0      0      0
         .1 .0460 .046 .050      0      .1     .25      0      0      0
         .1 .0460 .046 .050      0     -.1     .25      0      0      0
         .1 .0460 .023 .050   -.08   -.605       0      0      0      0
         .1 .0230 .023 .020      0   -.606       0      0      0      0
         .1 .0230 .046 .020    .06   -.605       0      0      0      0 ];

switch ptype
case {'shepp-logan', 'shepp-logan-zhu', 'zhu', ''}
	params = ezhu;
case {'shepp-logan-kak', 'kak'}
	params = ekak;
case {'shepp-logan-e3d', 'e3d'}
	params = e3d;
otherwise
	error('unknown parameter type %s', ptype)
end

params(:,[1 4]) = params(:,[1 4]) * xfov;
params(:,[2 5]) = params(:,[2 5]) * yfov;
params(:,[3 6]) = params(:,[3 6]) * zfov;
params(:,9) = params(:,8);
params(:,8) = 0; % z rotation
end % shepp_logan_3d_parameters()


%
% ellipsoid_im_profile()
%
function ellipsoid_im_profile
ig = image_geom('nx', 2^6, 'ny', 2^6-2, 'nz', 2^5, 'fov', 240, 'dz', 1);
ell = [30 20 2, 50 40 10, 20 0 100];
profile on
phantom = ellipsoid_im(ig, [], 'oversample', 2, 'fast', 0);
phantom = ellipsoid_im(ig, [], 'oversample', 2, 'fast', 1);
profile off
profile report

end % ellipsoid_im_profile()


%
% ellipsoid_im_test()
%
function ellipsoid_im_test

ig = image_geom('nx', 2^5, 'ny', 2^5-2', 'nz', 15, 'fov', 240, ...
	'dz', -6); % negative dz to match aspire
im pl 2 2
if 1
	phantom = ellipsoid_im(ig, [], 'oversample', 2);
	im(1, phantom, 'Shepp Logan', [0.9 1.1]), cbar
end

% compare to aspire
if ~has_aspire, return, end

ell = [30 20 10, 50 40 30, 20 0 100];
over = 2^1;

dir = test_dir;
file = [dir '/t.fld'];
com = 'echo y | op ellipsoid %s %d %d %d  %g %g %g  %g %g %g %g %g %d %d';
pix = [ig.dx -ig.dy -ig.dz ig.dx ig.dy -ig.dz 1 1 1];
com = sprintf(com, file, ig.nx, ig.ny, ig.nz, ell ./ pix, log2(over)+1);
os_run(com)
asp = fld_read(file);
im(4, ig.x, ig.y, asp, 'aspire'), cbar

%pr 4/3 * pi * prod(ell(4:6)) * ell(9)

for fast=1:-1:0
	mat = ellipsoid_im(ig, ell, 'oversample', over, 'fast', fast);
	t = sprintf('mat, fast=%d, z: %g to %g', fast, ig.z([1 end]));
	im(2, ig.x, ig.y, mat, t), cbar
	im(3, ig.x, ig.y, mat-asp, 'mat-aspire'), cbar
	max_percent_diff(mat, asp) % 25% different it seems

	if 1 % check centroid
		[xx yy zz] = ndgrid(ig.x, ig.y, ig.z);
		t = [sum(xx(:) .* mat(:)) sum(yy(:) .* mat(:)) ...
			sum(zz(:) .* mat(:))] / sum(mat(:));
		if any(abs(t - ell(1:3)) > 0.02), warn 'bad centroid', end
	end
%	pr sum(mat(:)) * abs(ig.dx * ig.dy * ig.dz);
if fast==1, prompt, end
end
end % ellipsoid_im_test()
