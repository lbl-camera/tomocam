 function xo = montager(xi, varargin)
%function xo = montager(xi, varargin)
% in
%	xi	[3d or 4d] set of images
% options
%	'col'		# of cols
%	'row'		# of rows
%	'aspect'	aspect ratio (default 1.2)
% out
%	xo [2d]		3d or 4d images arrange a as a 2d rectangular montage
% Copyright 1997, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if nargin == 1 & streq(xi, 'test'), montager_test, return, end

arg.aspect = 1.2; % trick: default 1.2/1 aspect ratio
arg.col = [];
arg.row = [];

arg = vararg_pair(arg, varargin);

if ndims(xi) > 4, warning('5d not done'), end
if ndims(xi) == 4
	[nx ny n3 n4] = size(xi);
	nz = n3*n4;
	xi = reshape(xi, [nx ny nz]);
else
	[nx ny nz] = size(xi);
end

if isempty(arg.col)
	if isempty(arg.row)
		if ndims(xi) == 4
			arg.col = n3;
		elseif nx == ny && nz == round(sqrt(nz)).^2 % perfect square
			arg.col = round(sqrt(nz));
		else
			arg.col = ceil(sqrt(nz * ny / nx * arg.aspect));
		end
	else
		arg.col = ceil(nz / arg.row);
	end
end

if isempty(arg.row)
	arg.row = ceil(nz / arg.col);
end

xo = zeros(nx * arg.col, ny * arg.row);

for iz=0:(nz-1)
	iy = floor(iz / arg.col);
	ix = iz - iy * arg.col;
	xo([1:nx]+ix*nx, [1:ny]+iy*ny) = xi(:,:,iz+1);
end


%
% montager_test()
%
function montager_test
t = [20 30 5];
t = reshape([1:prod(t)], t);
im pl 1 2
im(1, montager(t))
im(2, montager(t, 'row', 4))
