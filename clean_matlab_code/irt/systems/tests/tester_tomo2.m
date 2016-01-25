  function tester_tomo2(G1, mask, varargin)
%|function tester_tomo2(G1, mask, [options])
%| Test suite for a Fatrix-type 2D or 3D system object, including Gblock tests.
%|
%| option
%|	'G2'	Fatrix	optional 2nd object for testing and comparisons
%|	'multi'	0|1	see tester_tomo2()
%|
%| Copyright 2005-8-2, Jeff Fessler, University of Michigan

if nargin < 2, help(mfilename), error(mfilename), end

arg.multi = true;
arg.G2 = [];
arg = vararg_pair(arg, varargin);

Fatrix_test_basic(G1, mask, 'multi', arg.multi, ...
	'name', inputname(1), 'caller', caller_name)

switch ndims(mask)
case 2
	[nx ny] = size(mask);
	ig = image_geom('nx', nx, 'ny', ny, 'dx', 1);
	x = ellipse_im(ig, []);

case 3
	[nx ny nz] = size(mask);
	ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', 1);
	x = ellipsoid_im(ig, '');
end

nblock = 2;
tester_tomo2_block(G1, mask, x, nblock, arg.multi)

if ~isempty(arg.G2)
	Fatrix_test_basic(arg.G2, mask, 'multi', arg.multi, ...
		'name', inputname(4), 'caller', caller_name)
	tester_tomo2_compare(G1, arg.G2, x)
end


%
% tester_tomo2_compare()
%
function tester_tomo2_compare(G1, G2, x)

y1 = G1 * x;
y2 = G2 * x;
my_compare(y1, y2, 'G*x')

x1 = G1' * y1;
x2 = G2' * y1;
equivs(x1, x2)

j = round(size(G1,2) / 2); % roughly a middle pixel
y1 = G1(:,[j j+1]);
y2 = G2(:,[j j+1]);
my_compare(y1, y2, 'G(:,j)')

% check G(:,:)
if 0 & size(x,1) < 100
	t1 = G1(:,:);
	t2 = G2(:,:);
	my_compare(t1, t2, '(:,:)');
%	mpd = max_percent_diff(t1,t2);
%	printf('G(:,:)	mpd %g', mpd)
%	if mpd/100 > 1e-6, error 'G(:,:)', end
end

printm('passed %s', G1.caller)


%
% tester_tomo2_block()
% now block version
%
function tester_tomo2_block(G1, mask, x, nblock, multi)

B1 = Gblock(G1, nblock, 1);

% B*x
y1 = G1 * x;
y2 = B1 * x;
my_compare(y1, y2, 'B*x')

y0 = y1;
na = size(y0); na = na(end);

% B'*y
x1 = G1' * y0;
x2 = B1' * y0;
my_compare(x1, x2, 'B''*y')

%
% block operations
%
for k=1:nblock
	ia = k:nblock:na;
	str = sprintf('B{%d}', k);

	% check B{k}*x
	t1 = G1 * x;
	t1 = stackpick(t1,ia);
	t2 = B1{k} * x;
	my_compare(t1, t2, 'B{k}*x')

	% B{k} * [x x]
	if multi
		t2 = B1{k} * stackup(x,x);
		my_compare(stackup(t1,t1), t2, [str '*[x x]'])
	end

	% check B{k}*x(mask)
	t2 = B1{k} * x(mask);
	t2 = reshape(t2, size(t1));
	my_compare(t1, t2, [str 'x()'])

	% check B{k}*[x(mask) x(mask)]
	if multi
		t2 = B1{k} * [x(mask) x(mask)];
		my_compare([t1(:) t1(:)], t2, [str '[x() x()]'])
	end

	% check B{k}'*y()
	tmp = block_insert(ia, size(y0), stackpick(y0,ia));
	t1 = G1' * tmp(:);
	t2 = B1{k}' * col(stackpick(y0,ia));
%	my_compare(t1, t2, [str '''*y'])
	mpd = max_percent_diff(t1,t2);
	if mpd
		printf([str '''*y mpd %g'], mpd)
		if mpd/100 > 1e-6, error 'B{k}''*y()', end
	end
end


% my_compare()
function my_compare(t1, t2, arg)
%max_percent_diff(t1, t2, arg)
jf_equal(t1, t2)


% block_insert()
% for block or OS methods, make data array that is all zeros but at "ia"
function out = block_insert(ia, dims, data)
out = zeros(dims);
switch length(dims)
case 2
	out(:,ia) = data;
case 3
	out(:,:,ia) = data;
end
