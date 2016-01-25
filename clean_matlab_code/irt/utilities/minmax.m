 function r = minmax(x, varargin)
%function r = minmax(x, dim)
% Return mininum and maximum of values of input x.
% Default is to examine the entire array.
% Specify a dimension dim otherwise.
% If dim is a string, print...
% Copyright 2003, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end

dim = [];
if length(varargin) && isnumeric(varargin{1})
	dim = varargin{1};
	varargin = {varargin{2:end}};
end
str = '';
if length(varargin)
	if ischar(varargin{1})
		str = varargin{1};
	else
		error 'bug'
	end
end

if iscell(x)
	r = [];
	for ii = 1:length(x)
		if ~isempty(dim)
			r = [r, minmax(x{ii}, dim)];
		else
			r = [r, minmax(x{ii})];
	end
		end

elseif isstruct(x)
	if nargout == 0
		minmax_struct(x, inputname(1))
		return
	else
		r = [];
		names = fieldnames(x);
		for ii = 1:length(names)
			r = [r; minmax(x.(names{ii}))];
		end
	end

elseif ~isempty(dim)
	r = [min(x, [], dim); max(x, [], dim)];

else
	r = [min(x(:)); max(x(:))];
end

if issparse(r)
	r = full(r);
end

if nargout == 0
	if isempty(str) & ~isempty(inputname(1))
		str = [caller_name ': ' inputname(1) ':'];
	end

	if ncol(r) == 1
		if isempty(str)
%			disp(r.')
			printm('min=%g max=%g', r(1), r(2))
		else
			printf('%s min=%g max=%g', str, r(1), r(2))
		end
	else
		for ii=1:ncol(r)
			if isempty(str)
				printm('%d min=%g max=%g', ii, r(1,ii), r(2,ii))
			else
				printf('%s %d min=%g max=%g', str, ii, r(1,ii), r(2,ii))
			end
		end
	end
	clear r
end


%
% minmax_struct(st, prefix)
% show min/max of all elements of a structure, descending recursively.
%
function minmax_struct(st, prefix)
prefix = [prefix '.'];

names = fieldnames(st);
for ii=1:length(names)
	name = names{ii};
	t = st.(name);

	if isempty(t), continue, end

	if islogical(t) | isnumeric(t)
		if  numel(t) == 1
			printf('%s = %d', [prefix name], t)
		else
			printf('%s %g %g', ...
				[prefix name], min(abs(t(:))), max(abs(t(:))))
		end

	elseif isstruct(t)
		minmax_struct(t, [prefix name]);

	elseif ischar(t)
		printf([prefix name ' = "' t '"'])

	else
		printf('%s ?', [prefix name])
	end
end
