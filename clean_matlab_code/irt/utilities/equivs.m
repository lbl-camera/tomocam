 function out = equivs(var1, var2, varargin)
%function out = equivs(var1, var2, command)
% verify that var1 and var2 are equivalent to within single precision accuracy
% if not, print error message.  an alternative to isequal().
% See also: jf_equal
% Copyright 2007, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(var1, 'test'), equivs_test, return, end
if nargin < 2, help(mfilename), error(mfilename), end

arg.thresh = 1e-6;
arg = vararg_pair(arg, varargin);

if isempty(var1) && isempty(var2)
	ok = true;

elseif ~isequal(size(var1), size(var2))
		pr size(var1)
		pr size(var2)
		fail 'incompatible dimensions'

else
	var1 = var1(:);
	var2 = var2(:);
	norm = (max(abs(var1)) + max(abs(var2))) / 2;
	if ~norm
		ok = true; % both zero!
	else
		err = max(abs(var1-var2)) / norm;
		ok = err < arg.thresh;
	end
end

if nargout
	out = ok;
end

if ok
	return
end

[name line] = caller_name;
if isempty(name)
	str = '';
else
	str = sprintf('%s %d:', name, line);
end

minmax(var1)
minmax(var2)
diff = var1 - var2;
minmax(diff)
name1 = inputname(1);
name2 = inputname(2);
error([str ' normalized difference of %g between "%s" "%s"'], err, name1, name2)

function equivs_test
randn('state', 0)
x = randn(1000,200);
y = dsingle(x);
equivs(x,y)

passed = 0;
try
	y = x + 2e-6 * max(x(:));
	equivs(x,y)
	passed = 1;
catch
end
if passed, error 'this should failed!', end
