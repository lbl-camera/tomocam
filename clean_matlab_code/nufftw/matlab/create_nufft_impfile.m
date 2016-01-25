function create_nufft_impfile(varargin)
% CREATE_NUFFT_IMPFILE
% 	create_nufft_impfile(Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajectory, sqrtdcf, impfilename);
% 	create_nufft_impfile(Nthreads, imagesize, maxerr, alpha, resamplingmethod, trajfilename, impfilename);

% Michal Zarrouk, July 2013.

nufft_mex('create_impfile', varargin{:});

