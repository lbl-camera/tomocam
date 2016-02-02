 function [fwhm_best, costs, im_best] = ...
	fwhm_match(true_image, blurred_image, fwhms)
%function [fwhm_best, costs, im_best] = ...
%	fwhm_match(true_image, blurred_image, fwhms)
%
% given a blurred_image of a true_image, find the FHWM of a Gaussian kernel
% that, when convolved to the true_image, yields the smoothed image
% that best matches blurred_image.
%
% the set of FWHM values given in the array fwhms is tried.
%
% Copyright 2001-8-30, Jeff Fessler, The University of Michigan

%
% if no arguments, then run a simple test example
%
if ~nargin
	%	make a pyramidal test PSF to try to stress the approach
	psf1 = [0:5 4:-1:0]; psf1 = psf1 / sum(psf1); psf = psf1' * psf1;
	true_image = phantom('Modified Shepp-Logan', 128);
	blurred_image = conv2(true_image, psf, 'same');
	clf, im(221, true_image, 'True Image')
	im(222, blurred_image, 'Blurred Image')

	fwhms = [2:0.25:8];
	[fwhm_best, costs] = fwhm_match(true_image, blurred_image, fwhms);
	subplot(223), plot(fwhms, costs, 'c-o', fwhm_best, min(costs), 'yx')
	xlabel FWHM, ylabel Cost, title 'Cost vs FWHM' 
	np = length(psf);	ip = -(np-1)/2:(np-1)/2;
	kern = gaussian_kernel(fwhm_best);
	nk = length(kern);	ik = -(nk-1)/2:(nk-1)/2;
	subplot(224), plot(ip, psf1, '-o', ik, kern(:), '-+')
	xlabel pixel, title 'PSF profile: actual and Gaussian fit'

	help(mfilename)
return
end


if nargin < 3
	fwhms = 0:0.5:4;
end

costs = zeros(size(fwhms));
cost_min = Inf;
for ii=1:length(fwhms)
	fwhm = fwhms(ii);
	kern = gaussian_kernel(fwhm);
	psf = kern * kern';
	tmp = conv2(true_image, psf, 'same');
	costs(ii) = norm(tmp(:) - blurred_image(:)) / norm(true_image(:));
	if costs(ii) < cost_min
		im_best = tmp;
	end
end

[dummy, ibest] = min(costs);
if ibest == 1 | ibest == length(fwhms)
	warning 'need wider range of fwhms'
end
fwhm_best = fwhms(ibest);
