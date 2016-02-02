 function [dim, varargout] = size(ob, varargin)
%function dim = size(ob, varargin)
% or [dim1 dim2] = size(ob)
% "size" method for this class

% trick: handle block size carefully!
if ~isempty(ob.nblock) && ob.nblock > 1 && ~isempty(ob.iblock)
	dim = ob.dim;
	try
		na = ob.arg.nn(end);
	catch
		fail 'need to determine na!?'
	end
	ia = ob.iblock:ob.nblock:na;
	dim(1) = dim(1) / na * length(ia);

else % usual case (not block)
	dim = ob.dim;
end

if length(varargin) == 1
	dim = dim(varargin{1});
	return
elseif length(varargin) ~= 0
	error 'bug'
end

if nargout > 1
	varargout{1} = dim(2);
	dim = dim(1);
end
