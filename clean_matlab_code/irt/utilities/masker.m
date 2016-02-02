 function y = masker(x, mask)
%function y = masker(x, mask)
% extract from x the nonzero elements within (logical) mask
% in
%	x	[(Nd),(L)]	image(s)
%	mask	[(Nd)]		logical array, np = sum(mask)
% out
%	y	[np,(L)]	concise columns
%
% Copyright 2004-9-28, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(x, 'test'), masker_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end
if ~islogical(mask), error 'mask must be logical', end

xdim = size(mask);
L = size(x); L = L(1+ndims(mask):end);
x = reshape(x, [numel(mask) prod(L)]); % [*Nd,*L]
y = x(mask(:),:); % [np, *L]
y = reshape(y, [size(y,1) L]); % [np,(L)]


function masker_test
ig = image_geom('nx', 40, 'ny', 50, 'dx', 1);
ig.mask = [0 0 [ig.nx ig.ny]/2-5 0 1];
ig.mask = ellipse_im(ig) > 0;
ig.mask = conv2(double(ig.mask), ones(2), 'same') > 0;
x1 = ones([ig.nx ig.ny 3 2]);
y1 = masker(x1, ig.mask);
x2 = ig.embed(y1);
y2 = masker(x2, ig.mask);
if max_percent_diff(y1, y2), error 'bug', end
