 function y = reale(x, arg2, arg3)
%function y = reale(x)
%function y = reale(x, tol)
%function y = reale(x, 'warn', 'message')
%function y = reale(x, 'error')
%function y = reale(x, 'report')
%function y = reale(x, 'prompt')
%function y = reale(x, 'disp')
% return real part of complex data (with error checking).
% checks that imaginary part is negligible (or warning etc. if not)
% Copyright Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if nargin == 1 & streq(x, 'test'), reale_test, return, end

com = 'error';
if isa(x, 'double')
	tol = 1e-13;
else
	tol = 1e-6;
end

if nargin > 1
	if ischar(arg2)
		com = arg2;
	elseif isnumeric(arg2)
		tol = arg2;
	end
end

if streq(com, 'disp')
	;
elseif streq(com, 'warn')
	onlywarn = 1;
elseif streq(com, 'error')
	onlywarn = 0;
elseif streq(com, 'prompt')
	;
elseif streq(com, 'report')
	;
else
	error('bad argument %s', com)
end

max_abs_x = max(abs(x(:)));
if max_abs_x == 0
	if any(imag(x(:)) ~= 0)
		error 'max real 0, but imaginary!'
	else
		y = real(x);
		return
	end
end

frac = max(abs(imag(x(:)))) / max_abs_x;
if streq(com, 'report')
	printm('imaginary part %g%%', frac * 100)
	return
end

if frac > tol
	[cname line] = caller_name;
	t = sprintf('%s(%d): %s: imaginary fraction of %s is %g', ...
		cname, line, mfilename, inputname(1), frac);
	if isvar('arg3')
		t = [t ', ' arg3];
	end
	if streq(com, 'disp')
		disp(t)

	elseif streq(com, 'prompt')
		printm('reale() called for input with imaginary part %g%%', frac * 100)
		printm('reale() called in context where a large imaginary part')
		printm('is likely an *error*.  proceed with caution!')
		t = input('proceed? [y|n]: ', 's');
		if isempty(t) || t(1) ~= 'y'
			printm('ok, aborting is probably wise!')
			error ' '
		end

	elseif onlywarn
		disp(t)
	else
		error(t)
	end
end
y = real(x);


function reale_test
x = 7 + 1i*eps;
reale(x, 'warn');
reale(x, 'prompt');
reale(x, 'report');
%reale(x, eps/100) % check error reporting
