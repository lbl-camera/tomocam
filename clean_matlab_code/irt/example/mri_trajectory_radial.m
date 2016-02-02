function [omega wi] = mri_trajectory_radial(N, fov, varargin)
arg.na_nr = 2*pi;	% default ensures proper sampling at edge of k-space
arg.na = [];		% angular spokes (default: na_nr * nr)
arg.nr = max(N)/2;	% radial samples per spoke
arg.ir = [];		% default: 0:nr
arg.omax = pi;		% maximum omega
arg = vararg_pair(arg, varargin);
if isempty(arg.ir), arg.ir = [0:arg.nr]; end
if isempty(arg.na), arg.na = 4*ceil(arg.na_nr * arg.nr/4); end % mult of 4
om = arg.ir/arg.nr * pi;
ang = [0:arg.na-1]/arg.na * 2*pi;
[om ang] = ndgrid(om, ang); % [nr+1, na]
omega = [col(om.*cos(ang)) col(om.*sin(ang))];

% density compensation factors based on "analytical" voronoi
if any(fov ~= fov(1)), fail('only square FOV implemented for radial'), end
du = 1/fov(1); % assume this radial sample spacing
wi = pi * du^2 / arg.na * 2 * arg.ir(:); % see lauzon:96:eop, joseph:98:sei
wi(arg.ir == 0) = pi * (du/2)^2 / arg.na; % area of center disk
wi = repmat(wi, [1 arg.na]);
wi = wi(:);