 function h = titlef(varargin)
%|function h = titlef(varargin)
%| version of title with built-in sprintf

if isfreemat
	for ii=1:length(varargin)
		if streq(varargin{ii}, 'interpreter') % not supported by freemat
			varargin{ii} = {};
			varargin{ii+1} = {};
		end
	end
end

hh = title(sprintf(varargin{:}));
if nargout
	h = hh;
end
