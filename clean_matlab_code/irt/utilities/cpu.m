 function out = cpu(arg, varargin)
%function out = cpu(arg, varargin)
% cpu tic
% cpu toc
% cpu toc printarg
% work like tic/toc except they use cpu time instead of wall time.
% also:
% cpu etic
% cpu etoc
% for elapsed time

if nargin < 1, help(mfilename), error(mfilename), end
if streq(arg, 'test'), cpu_test, return, end

% todo: add default behaviour

persistent t
if ~isvar('t') || isempty(t)
	t.tic = [];
	t.clock = [];
end

if streq(arg, 'tic')
	t.tic = cputime;
	if length(varargin), error 'tic takes no option', end
	return

elseif streq(arg, 'etic')
	t.clock = clock;
	if length(varargin), error 'tic takes no option', end
	return
end

if streq(arg, 'toc')
	if isempty(t.tic)
		error 'must initialize cpu with tic first'
	end
	out = cputime - t.tic;

elseif streq(arg, 'etoc')
	if isempty(t.clock)
		error 'must initialize cpu with etic first'
	end
	out = etime(clock, t.clock);

else
	error 'bad argument'
end

if length(varargin) == 1
	arg = varargin{1};
	if ischar(arg)
		printf('%s %g', arg, out)
	end

	if ~nargout
		clear out
	end
elseif length(varargin) > 1
	error 'too many arguments'
end


function cpu_test
cpu tic
cpu toc
cpu toc hello
cpu etic
cpu etoc
cpu etoc goodbye
out = cpu('etoc', 'test');
