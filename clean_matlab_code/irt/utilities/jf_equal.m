 function jf_equal(a, b)
%function jf_equal(a, b)
% verify that the two arguments are equal.
% if not, print error message.
% See also: equivs
% note: to compare two (e.g. Fatrix) objects, use jf_equal(struct(a), struct(b))
% Copyright 2007, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if nargin == 1 && streq(a, 'test'), jf_equal_test, return, end

if isequal(a, b), return, end
	
[name line] = caller_name;
if isempty(name)
	str = '';
else
	str = sprintf('%s %d', name, line);
end

if streq(class(a), 'strum')
	a = struct(a);
end

if streq(class(b), 'strum')
	b = struct(b);
end

if isstruct(a) && isstruct(b)
	jf_compare_struct(a, b)
else
	minmax(a)
	minmax(b)
	max_percent_diff(a, b)
end

aname = inputname(1);
bname = inputname(2);
fail([str ': "%s" and "%s" unequal'], aname, bname)


function jf_compare_struct(s1, s2)
%[a perm] = orderfields(a);
try
	s2 = orderfields(s2, s1);
catch
	warn 'different fields'
	return
end
names = fieldnames(s1);
for ii=1:length(names)
	name = names{ii};
	f1 = getfield(s1, name);
	f2 = getfield(s2, name);
	try
		jf_equal(f1, f2)
	catch
		warn('field %s differs', name)
	end
end


function jf_equal_test
a = 7;
b = 7;
c = 8;
jf_equal(a,b)
jf_equal(a,7)
%jf_equal(a,c)
