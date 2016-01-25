 function dd = penalty_displace(offsets, sizes)
%function dd = penalty_displace(offsets, sizes)
%
% convert scalar offsets to vector displacements, i.e.,
% find dx,dy such that dx*1 + dy*nx + dz*nx*ny + ... = offset.
% e.g., if offset = nx, then [dx dy dz] = [0 1 0]
% in
%	offsets	[LL,1]
%	sizes	[1,ndim]
% out
%	dd	[LL,ndim]
%
% Copyright 2006-12-6, Jeff Fessler, The University of Michigan
if nargin == 1 && streq(offsets, 'test'), penalty_displace_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

ndim = length(sizes);
half = max(floor(sizes/2), 1); % upper bound on dx is almost half the size

displace = zeros(length(offsets), ndim);

if 0 % this simpler way does not work...
	for id=ndim:-1:1
		nd = prod(sizes(1:id-1));
		sd = sizes(id);
	%	dis(id) = rem(offset + 0*half(id) * nd, nd) - 0*half(id);
		dd(:,id) = floor((offsets + half(id) * nd) / nd) - half(id);
		offsets = offsets - dd(:,id) * nd;
	end
return
end

for ll=1:length(offsets)
	offset = double(offsets(ll)); % trick: necessary!
	subval = 0;
	dis = zeros(1,ndim);
	for id=ndim:-1:2
		tmp = offset + sum((half(1:id-1)-1) .* [1 sizes(1:id-2)]);
		tmp = tmp + prod(sizes(1:id)) - subval;
		dis(id) = floor(tmp / prod(sizes(1:id-1))) - sizes(id);
		subval = subval + dis(id) * prod(sizes(1:id-1));
	end
	dis(1) = offset - subval;

	jf_assert all(abs(dis) < half)
	dd(ll,:) = dis;
end

if any(dd * [1 cumprod(sizes(1:end-1))]' ~= offsets(:))
	error 'bug'
end
end % penalty_displace


%
% penalty_displace_test()
%
function penalty_displace_test

nx = 100; ny = 80; % 2d
[ix iy] = ndgrid(-2:2, -2:2);
offsets = col(ix + iy * nx);
dd = penalty_displace(offsets, [nx ny]);
jf_equal(dd, [ix(:) iy(:)])

nx = 10; ny = 8; nz = 7; % 3d
[ix iy iz] = ndgrid(-2:2, -2:2, -2:2);
offsets = col(ix + iy * nx + iz * nx * ny);
dd = penalty_displace(offsets, [nx ny nz]);
jf_equal(dd, [ix(:) iy(:) iz(:)])

printm 'ok'
end % penalty_displace_test()
