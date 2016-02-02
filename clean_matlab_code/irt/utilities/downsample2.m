 function y = downsample2(x, m)
%function y = downsample2(x, m)
% downsample by averaging by integer factors
% m can be a scalar (same factor for both dimensions)
% or a 2-vector

if nargin == 1 & streq(x, 'test'), downsample2_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

y = downsample1(x, m(1));
y = downsample1(y', m(end))';


% 1d down sampling of each column
function y = downsample1(x, m)
n1 = floor(size(x,1) / m);
n2 = size(x,2);
y = zeros(n1,n2);
for ii=0:m-1
	y = y + x(ii+[1:m:m*n1],:);
	ticker(mfilename, ii+1,m)
end
y = y / m;


function downsample2_test

x = reshape(1:24, [4 6])
y = downsample2(x, 2)
