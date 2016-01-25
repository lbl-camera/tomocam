 function smap = mri_sensemap_sim(varargin)
%function smap = mri_sensemap_sim(varargin)
%
% Simulate sensitivity maps for sensitivity-encoded MRI
% based on wang:00:dop
% option
%	nx, ny, dx, dy, ncoil, rcoil, orbit (see below)
% out:
%	smap	[nx,ny,ncoil]	simulated sensitivity maps
%
% todo: currently only returns a *magnitude* map.  in fact the smap
% should be complex, and perhaps grivich:00:tmf gives the formulae needed.
%
% Copyright 2005-6-20, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(varargin{1}, 'test'), mri_sensemap_sim_test, return, end

arg.chat = true;
arg.nx = 64;
arg.ny = [];
arg.dx = 3; % pixel size in mm
arg.dy = [];
arg.ncoil = 4; % # of coils
arg.rcoil = 100; % coil radius
arg.orbit = 360;

arg = vararg_pair(arg, varargin);

if isempty(arg.dy), arg.dy = arg.dx; end
if isempty(arg.ny), arg.ny = arg.nx; end

smap = mri_sensemap_sim_do(arg.nx, arg.ny, arg.dx, arg.dy, ...
	arg.ncoil, arg.rcoil, arg.orbit, arg.chat);

%
% mri_sensemap_sim_do()
%
function smap = mri_sensemap_sim_do(nx, ny, dx, dy, ncoil, rcoil, orbit, chat)

rlist = rcoil * ones(ncoil, 1); % coil radii

plist = zeros(ncoil,3); % position of coil center
nlist = zeros(ncoil,3); % normal vector (inward) from coil center

% circular coil configuration, like head coils
for ii=1:ncoil
	phi = deg2rad(orbit)/ncoil * (ii-1);
	Rad = nx/2 * dx * 1.2;
	plist(ii,:) = Rad * [cos(phi) sin(phi) 0];
	nlist(ii,:) = -[cos(phi) sin(phi) 0];
end

x = ([1:nx]-(nx+1)/2)*dx;
y = ([1:ny]-(ny+1)/2)*dy;
z = 0;
[xx yy zz] = ndgrid(x,y,z);

smap = zeros(nx, ny, ncoil);
for ii=1:ncoil
	% rotate coordinates to correspond to coil orientation
	zr =	(xx - plist(ii,1)) .* nlist(ii,1) + ...
		(yy - plist(ii,2)) .* nlist(ii,2) + ...
		(zz - plist(ii,3)) .* nlist(ii,3);
	xr =	xx .* nlist(ii,2) - yy .* nlist(ii,1);
	smap(:,:,ii) = mri_smap1(xr, 0, zr, rlist(ii));
end
smap = smap * rlist(1) / (2*pi); % trick: scale so near unity maximum

% plot array geometry in z=0 plane
if chat && im
	im clf; im pl 2 2
	for ii=1:ncoil
		clim = [0 max(smap(:))];
		im(ii, x, y, abs(smap(:,:,ii)), clim), cbar
%		xmax = max(max(abs(x)), max(plist(:,1)));
%		ymax = max(max(abs(y)), max(plist(:,2)));
		xmax = max([max(abs(x)) max(abs(y)) max(col(plist(:,[1 2])))]);
%		axis([-xmax xmax -ymax ymax]*1.05)
		axis(xmax*[-1 1 -1 1]*1.1)

		hold on
		plot(0,0,'.', plist(:,1), plist(:,2), 'o')
		xdir = nlist(ii,2);
		ydir = nlist(ii,1);
		r = rlist(ii);
		plot(plist(ii,1)+r*xdir*[-1 1], plist(ii,2)+r*ydir*[1 -1], '-')
		hold off
	end
end

if ~nargout, clear smap, end

%
% based on wang:00:dip
% for a circular coil in x-y plane of radius a
%
function smap = mri_smap1(x, y, z, a)
x = x ./ a;
y = y ./ a;
z = z ./ a;
r = sqrt(x.^2 + y.^2);
M = 4 * r ./ ((1 + r).^2 + z.^2);
[K E] = ellipke(M);
smap = 2*((1+r).^2 + z.^2).^(-0.5) .* ...
	(K + (1 - r.^2 - z.^2) ./ ((1-r).^2 + z.^2) .* E);
smap = smap / a;

%m = linspace(0,1,201);
%m = 0;
%[k e] = ellipke(m)
%plot(m, k, '-', m, e, '--')


%
% mri_sensemap_sim_test
%
function mri_sensemap_sim_test
mri_sensemap_sim('chat', 1);
