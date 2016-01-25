 function centers = highrate_centers(data, L, M)
%function centers = highrate_centers(data, L, M)
% According to high-rate scalar quantization theory (Gersho, T-IT, 1979),
% the density of MMSE quantization cell centroids should be f^{k/(k+2)},
% for k-dimensional data distributed with pdf f.
% This m-file designs L centroids for scalar data (k=1) using this principle.
% M is the number of histogram bins used to approximate the data pdf. 
% out: centers is [L,1]
% Copyright 2004-7-7, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if ~isvar('M'), M=100; end

data = data(:);
[wt data] = hist(data, M);
dens = wt .^ (1/3); % from high-rate scalar quantization theory
cdf = cumsum(dens / sum(dens));
[cdf ii] = unique(cdf);
m1 =  [1:M]-0.5;
m1 = m1(ii);
uu = ([1:L]-0.5)/L;
m2 = interp1(cdf, m1, uu, 'cubic');
%plot(cdf, m1, '.', uu, m2, 'o')
centers = data(round(m2));
