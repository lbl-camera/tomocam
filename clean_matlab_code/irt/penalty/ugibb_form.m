 function ugibb = ugibb_form(pic, varargin)
%function ugibb = ugibb_form(pic, [options])
% options
%	'threshold'
%	'offset'
% form a "ugibb" structure from a 2D or 3D picture

if nargin < 1, help(mfilename), error(mfilename), end
if streq(pic, 'test'), ugibb_test, return, end

arg.threshold = 0;
arg.offsets = [];
arg = vararg_pair(arg, varargin);

[nx ny nz] = size(pic);

% 2d case
if nz==1
	if isempty(arg.offsets), arg.offsets = [1 nx nx+1 nx-1]; end
	ii = (nx+2):(nx*ny);	% trick: no dx on 1st image row...

	if 1
		dd = zeros(nx*ny, length(arg.offsets));
		for io=1:length(arg.offsets)
			off = arg.offsets(io);
%			dd(:,ii) = pic(ii) - pic(ii-off);
			ii = (1+off):(nx*ny);
			dd(ii,io) = pic(ii) - pic(ii-off);
		end
		dd = reshape(dd, nx, ny, length(arg.offsets));
	else
		dx = zeros(nx,ny); dy = dx; dp = dx; dn = dx;
		dx(ii) = pic(ii) - pic(ii-1);
		dy(ii) = pic(ii) - pic(ii-nx);
		dp(ii) = pic(ii) - pic(ii-nx+1);
		dn(ii) = pic(ii) - pic(ii-nx-1);
		dd(:,:,1) = dx;
		dd(:,:,2) = dy;
		dd(:,:,3) = dp;
		dd(:,:,4) = dn;

	end
	ugibb = double6(abs(dd) < arg.threshold);

	% fix the borders (where no neighbors exist)
	if 1
		ugibb(1,:,[1 4]) = 0;
		ugibb(:,1,[2 3 4]) = 0;
		ugibb(end,:,3) = 0;
	end

%	ugibb(:,:,[3 4]) = ugibb(:,:,[3 4]) / sqrt(2);

%
% 3d case
% with careful treatment of edges so zero weight there!
%
else
	if isempty(arg.offsets)
		arg.offsets = [1 nx nx+1 nx-1 nx*ny];
	end
	dx = zeros(nx,ny,nz,'single') + inf; dy = dx; dp = dx; dn = dx; dz = dx;

	ix = col([2:nx]'*ones(1,ny*nz) + ones(nx-1,1)*([1:ny*nz]-1)*nx);
	dx(ix) = pic(ix) - pic(ix-1);

	[ix iy iz] = ndgrid(1:nx,2:ny,1:nz);
	ii = ix + ((iz-1)*ny+(iy-1))*nx;
	dy(ii) = pic(ii) - pic(ii-nx);

	[ix iy iz] = ndgrid(1:nx-1,2:ny,1:nz);
	ii = ix + ((iz-1)*ny+(iy-1))*nx;
	dp(ii) = pic(ii) - pic(ii-nx+1);

	[ix iy iz] = ndgrid(2:nx,2:ny,1:nz);
	ii = ix + ((iz-1)*ny+(iy-1))*nx;
	dn(ii) = pic(ii) - pic(ii-nx-1);

	iz = (nx*ny+1):(nx*ny*nz);
	dz(iz) = pic(iz) - pic(iz-nx*ny);

	if 0 % old way
		dd = zeros(nx,ny,5,nz);
		dd(:,:,1,:) = dx;
		dd(:,:,2,:) = dy;
		dd(:,:,3,:) = dp;
		dd(:,:,4,:) = dn;
		dd(:,:,5,:) = dz;
	else
		dd = cat(4, dx, dy, dp, dn, dz);
	end
	ugibb = abs(dd) < arg.threshold;

	% diagonals
	if 0 % old
		ugibb(:,:,[3 4],:) = ugibb(:,:,[3 4],:) / sqrt(2);
	else
		ugibb(:,:,:,[3 4]) = ugibb(:,:,:,[3 4]) / sqrt(2);
	end
end


if 0
	rot180 = inline('rot90(x,2)');
	dx = flipud(conv2(flipud(pic), [1 -1]', 'same'));
	dy = fliplr(conv2(fliplr(pic), [1 -1], 'same'));
	dn = rot180(conv2(rot180(pic), [1 0; 0 -1], 'same'));
	dp = fliplr(conv2(fliplr(pic), [1 0; 0 -1], 'same'));

	da = dx;	% all 4 of them
	da(:,:,2) = dy;
	da(:,:,3) = dp;
	da(:,:,4) = dn;
	u = abs(da) < arg.threshold;
end

%
% ugibb_test.m
%
% test ugibb_form() and ugibb_sum()
function ugibb_test

if 0 % 2d
	pic = zeros(7,5);
	pic(3,2) = 1;
	pic = zeros(8,10);
	pic(4:6,4:6) = 1;
	[nx,ny] = size(pic);
else % 3d
	pic = zeros(8,10,6);
	pic(4:6,4:6,2:3) = 1;
	pic(end/2,end/2,end) = 1;
end

u = ugibb_form(pic, 'threshold', 0.02);
s = ugibb_sum(u);

im clf, im pl 2 4
p = @(ii) stackpick(u, ii);
im(2, p(1), 'wx')
im(3, p(2), 'wy')
im(6, p(3), 'wp')
im(7, p(4), 'wn')
im(8, p(5), 'wz')
im(1, pic, 'pic')
im(5, s, 'sum')
