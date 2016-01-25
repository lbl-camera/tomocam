  function [fw, angle, rad] = fwhm2(image, varargin)
%|function [fw, angle, rad] = fwhm2(image, [options])
%| in
%|	image	[nx,ny]	psf
%| option
%|	dx		pixel size (default: 1)
%|	dy		pixel size (default: dx)
%|	level		default: 0.5
%|	chat
%| out
%|	fw
%|
%| compute 2d fwhm of point-spread function
%| uses image maximum as center and the contourc function

% todo: what if image is complex?
% todo: option for user-specified peak?

if nargin < 1, help(mfilename), error(mfilename), end
if streq(image, 'test'), fwhm2_test, return, end

arg.dx = 1;
arg.dy = [];
arg.chat = (nargout == 0);
arg.level = 0.5;

if nargin > 1 && isnumeric(varargin{1}) % old style: dx, chat
	arg.dx = varargin{1}; varargin = {varargin{2:end}};
	if length(varargin) > 1
		arg.chat = varargin{2}; varargin = {varargin{2:end}};
	end
end

arg = vararg_pair(arg, varargin);
if isempty(arg.dy), arg.dy = arg.dx; end

% image maximum
if min(size(image)) < 11
	image = padn(image, max(size(image), 23));
end
[nx ny] = size(image);
ii = imax(image, 2);

% find better center estimate by local centroid
cx = ii(1);
cy = ii(2);
if 1
	ix = [-5:5]';
	iy = [-5:5]';
	t = image(cx + ix, cy + iy);
	if arg.chat
		im(ix, iy, t)
	end
	o.x = 0; o.y = 0;
	o.x = sum(t,2)'/sum(t(:)) * ix;
	o.y = sum(t,1)/sum(t(:)) * iy;
	cx = cx + o.x;
	cy = cy + o.y;
	if 0
		hold on
		plot(o.x, o.y, 'rx')
		hold off
	return
	end
end

image = double(image); % stupid matlab
cc = contourc(image, [1e30 arg.level * max(image(:))]);
if isempty(cc), error 'empty contour?  check minimum!', end
cc = cc(:,2:length(cc))';
cc = cc(:,[2 1]);	% swap row,col or x,y

% check center pixel found
if arg.chat && im
	clf
	im(121, image)
	hold on
	plot(cc(:,1), cc(:,2), '+')
	plot(cx, cy, 'rx')
	title(sprintf('length(cc)=%d', length(cc)))
	hold off
%	axis([40 60 50 70])
%	prompt
end

xx = arg.dx * (cc(:,1) - cx); % physical coordinates
yy = arg.dy * (cc(:,2) - cy);
rsamp = sqrt(xx.^2 + yy.^2);
tsamp = rad2deg(atan2(yy,xx));

angle = [0:180]';
r1 = interp1x(tsamp, rsamp, angle);
r2 = interp1x(tsamp, rsamp, angle-180);
% plot(tsamp, rsamp, 'o', angle, r1, '-', angle-180, r2, '-')
rad = r1 + r2;
fw = mean(rad); % fix: should the "average" be done s.t. for an ellipse
		% contour we get the avg of the two major axes?

if arg.chat && im
	subplot(122)
	plot(angle, rad, '-o')
	xlabel 'Angle [degrees]'
	ylabel 'FWHM [pixels]'
	zoom on
end


%
% fwhm2_test
%
function fwhm2_test
dx = 3;
dy = 2;
nx = 100;
ny = 80;
x = [-nx/2:nx/2-1]' * dx;
y = [-ny/2:ny/2-1]' * dy;
fx = 30;
fy = 16;
sx = fx / sqrt(log(256));
sy = fy / sqrt(log(256));
[xx yy] = ndgrid(x,y);
psf = exp(-((xx/sx).^2 + (yy/sy).^2)/2);
im pl 1 2
im(1, x, y, psf, 'psf'), axis equal, axis image
[fw ang rad] = fwhm2(psf, 'dx', dx, 'dy', dy);
im subplot 2
plot(ang, rad, '-o'), ylabel 'fwhm(\phi) [mm]'
axisx(0,180), xtick([0 90 180]), xlabel '\phi [degrees]'
titlef('overall fwhm=%g', fw)
