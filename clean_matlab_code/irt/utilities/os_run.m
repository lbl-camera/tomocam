 function out = os_run(str)
%function out = os_run(str)
% call OS (unix of course), check for error, optionally return output

[s out1] = unix(str);
if s
	error(sprintf('unix call failed:\n%s', str))
end

if nargout
	out = out1;
end
