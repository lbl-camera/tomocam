  function pr(varargin)
%|function pr(command)
%| print a message with the calling routine's name,
%| the argument that is evaluated, and the value thereof.

if nargin < 1, help(mfilename), error(mfilename), end

arg = [varargin{:}]; % handle cases like 'pr x + y'

tmp = evalin('caller', arg);

% scalar:
if isscalar(tmp) && isnumeric(tmp) && isreal(tmp)
	printf('%s: %s = %g', caller_name, arg, tmp)

% short vector:
elseif min(size(tmp)) == 1 && length(tmp) <= 3 && isnumeric(tmp) && isreal(tmp)
	printf('%s: %s = %s', caller_name, arg, num2str(tmp(:)', ' %g'))

% string:
elseif ischar(tmp)
	printf('%s: %s = %s', caller_name, arg, tmp)

% array, etc.:
else
	printf('%s: %s =', caller_name, arg)
	disp(tmp)
end
