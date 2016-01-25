  function out = jf_mip3(vol, varargin)
%|function out = jf_mip3(vol, varargin)
%|
%| create a MIP (maximum intensity projection) mosaic from a 3D volume
%| in	
%|	vol	[]	[nx ny nz] 3d	
%| option
%|	show	bool	default: 1 if no output, 0 else
%| out
%|	out	[]	[nx+nz ny+nz] 2d	
%|
%| if no output, then display it using im()

if nargin < 1, help(mfilename), error(mfilename), end
if nargin == 1 && streq(vol, 'test'), jf_mip3_test, return, end

arg.show = ~nargout;
arg = vararg_pair(arg, varargin);

if ndims(vol) ~= 3, fail '3d only', end

vol = abs(vol);
xy = max(vol, [], 3); % [nx ny]
xz = max(vol, [], 2); % [nx nz]
yz = max(vol, [], 1); % [ny nz]
xz = squeeze(xz);
zy = squeeze(yz)';
nz = size(vol,3);

out = [	xy, xz;
	zy, zeros(nz,nz)];

if arg.show
	im(out)
end

if ~nargout
	clear out
end

function jf_mip3_test
ig = image_geom('nx', 64, 'ny', 60, 'nz', 32, 'dx', 1);
vol = ellipsoid_im(ig, [0 0 0 ig.nx/3 ig.ny/4 ig.nz/3 20 0 1]);
mip = jf_mip3(vol);
%im(vol), prompt
im(mip)
