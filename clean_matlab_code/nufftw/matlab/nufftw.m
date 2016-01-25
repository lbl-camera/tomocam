function varargout = nufftw(varargin)
% tune and write to file
% nufftw(Nthreads, imagesize, maxerr, resamplingmethod, trajfilename, tuning_heuristic, alphastart, alphaend, nalpha, impfilename); (10)
% nufftw(Nthreads, imagesize, maxerr, resamplingmethod, trajectory, dcf, tuning_heuristic, alphastart, alphaend, nalpha, impfilename); (11)

% tune and return imp
% imp = nufftw(Nthreads, imagesize, maxerr, resamplingmethod, trajfilename, tuning_heuristic, alphastart, alphaend, nalpha); (9)
% imp = nufftw(Nthreads, imagesize, maxerr, resamplingmethod, trajectory, dcf, tuning_heuristic, alphastart, alphaend, nalpha); (10)

% Michal Zarrouk, July 2013.


