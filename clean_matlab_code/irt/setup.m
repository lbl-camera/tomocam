% setup.m
% run this file to set up matlab path etc.
% you may need to modify this depending on how you installed the toolbox
% so this should be considered simply a "guide" not a robust script.

if ~exist('irtdir', 'var')
	disp('The variable "irtdir" is not set, so trying default, assuming')
	disp('that you launched matlab from the irt install directory.')
	disp('You may need to edit setup.m or adjust your path otherwise.')

%	irtdir = pwd; % the default is to assume launch from irt directory

	% default is to look for directory where this setup.m is installed!
	irtdir = which('setup'); % find setup.m
	[irtdir dummy] = fileparts(irtdir);
	clear dummy

	disp(['Assuming you installed irt in directory "' irtdir '".'])

%	irtdir = '~fessler/l/src/matlab/alg/'; % where you install this package
%	irtdir = '~fessler/l/web/irt/'; % where you install this package
end

if ~exist(irtdir, 'dir')
	disp(sprintf('The directory "%s" does not exist', irtdir))
	error(sprintf('you need to edit %s to change default path', mfilename))
end

if irtdir(end) ~= filesep % make sure there is a '/' at end of directory
	irtdir = [irtdir filesep];
end

path([irtdir 'align'], path);		% image registration
path([irtdir 'align/mex'], path);	% image registration mex files
path([irtdir 'blob'], path);		% blob (KB) basis
path([irtdir 'ct'], path);		% x-ray CT (polyenergetic) recon
path([irtdir 'data'], path);		% example data
path([irtdir 'emission'], path);	% emission image reconstruction
path([irtdir 'example'], path);		% example applications
path([irtdir 'fbp'], path);		% FBP (filtered backprojection) code
path([irtdir 'general'], path);		% generic image reconstruction
path([irtdir 'graph'], path);		% graphics routines
path([irtdir 'mri'], path);		% MRI reconstruction
path([irtdir 'mri-rf'], path);		% MRI RF pulse design
%path([irtdir 'mri/recon'], path);	% MRI reconstruction - old
path([irtdir 'nufft'], path);		% nonuniform FFT (for a fast projector)
path([irtdir 'nufft/table'], path);	% mex files for NUFFT
path([irtdir 'penalty'], path);		% regularizing penalty functions
path([irtdir 'systems'], path);		% system "matrices"
path([irtdir 'systems/tests'], path);	% tests of systems
path([irtdir 'transmission'], path);	% transmission image reconstruction
path([irtdir 'utilities'], path);	% various utility functions
path([irtdir 'wls'], path);		% weighted least-squares (WLS) estimates

if isempty(which('dbstack')) % for freemat only!
	path([irtdir 'freemat'], path);	% extra stuff for freemat only!
end

% Set up path to mex files, possibly depending on matlab version.
% Fortunately it seems that the v6 ones will run on v7 too.
% Unfortunately, it seems that sometimes compiling on one version
% e.g., 7.3 and running on earlier 7.0 won't work.
% if you have this problem, then comment out the mex path line(s) below.
% Much of the toolbox will work without mex, just slower.
%if str2num(version('-release')) <= 13
%	path([irtdir 'mex/v6'], path)
%else
	path([irtdir 'mex/v7'], path);
%end

% check to see if path setup worked by looking for im() routine.
if strcmp([irtdir 'graph' filesep 'im.m'], which('im'))
	disp('Path setup for irt appears to have succeeded.')
else
	disp('Path setup for irt may have failed.')
end
