  function im_toggle(i1, i2, varargin)
%|function im_toggle(i1, i2, varargin)
%| toggle between two images via keypress

if nargin < 2, help(mfilename), error(mfilename), end

% toggle between two images

while (im)
	im clf
	im(i1, varargin{:})
	fig_text(0.01, 0.01, ['toggle i1: ' inputname(1)])

%	pause
	in = input('hit enter for next image, or "q" to quit ', 's');
	if streq(in, 'q'), break, end

	im clf
	im(i2, varargin{:})
	fig_text(0.99, 0.01, ['toggle i2: ' inputname(2)], {'horiz', 'right'})

%	pause
	in = input('hit enter for next image, or "q" to quit ', 's');
	if streq(in, 'q'), break, end
end

set(gca, 'nextplot', 'replace')
