 function y = downsample3(x, m)
%function y = downsample3(x, m)
% downsample by averaging by integer factors
% function modified to 3D space by: Taka Masuda

if nargin == 1 & streq(x, 'test'), downsample3_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

if length(m) == 1
	m = m * ones(ndims(x),1);
end
if length(m) ~= ndims(x), error 'bad m', end

% downsample along each dimension
y = downsample1(x, m(1));
y = permute(downsample1(permute(y,[2 1 3]), m(2)), [2 1 3]);
y = permute(downsample1(permute(y,[3 2 1]), m(3)), [3 2 1]);


%
% 1d down sampling of each column
%
function y = downsample1(x, m)
if m == 1, y = x; return; end
n1 = floor(size(x,1) / m);
n2 = size(x,2);
n3 = size(x,3);
y = zeros(n1,n2,n3);
for ii=0:m-1
	y = y + x(ii+[1:m:m*n1],:,:);
	ticker(mfilename, ii+1, m)
end
y = y / m;


%
% downsample3_test
%
function downsample3_test
x = [6 5 2];
x = reshape(1:prod(x), x)
y = downsample3(x, 2)

if has_aspire
	filex = [test_dir filesep 'testx.fld'];
	filey = [test_dir filesep 'testy.fld'];
	fld_write(filex, x)
	delete(filey)
	com = ['op sample3 mean ' filey ' ' filex ' 2 2 2'];
	os_run(com)
	z = fld_read(filey);
	if ~isequal(y, z), error 'aspire/matlab mismatch', end
end
