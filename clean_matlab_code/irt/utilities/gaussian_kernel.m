 function kern = gaussian_kernel(fwhm, nk_half)
%function kern = gaussian_kernel(fwhm, nk_half)
% samples of a gaussian kernel at [-nk_half:nk_half]
% with given FWHM in pixels
% uses integral over each sample bin so that sum is very close to unity
%
% Copyright 2001-9-18, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), return, end
if nargin < 2, nk_half = 2 * ceil(fwhm); end

if fwhm == 0
	kern = zeros(nk_half*2+1, 1);
	kern(nk_half+1) = 1;
else
	sig = fwhm / sqrt(log(256));
	x = [-nk_half:nk_half]';
	kern = normcdf(x+1/2, 0, sig) - normcdf(x-1/2, 0, sig);
end
