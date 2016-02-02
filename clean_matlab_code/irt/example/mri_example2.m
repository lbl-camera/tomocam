% mri_example2.m
% Examples illustrating regularized iterative reconstruction for MRI
% from nonuniform k-space samples.
% (These examples do not include field inhomogeneity or relaxation.)
% More generally, this shows how to go from nonuniform samples
% in the frequency domain back to uniform samples in the space domain
% by an interative algorithm.
% Copyright 2004-4-20, Jeff Fessler, The University of Michigan

%
% inline functions for true object and its Fourier space
%
if ~isvar('xtrue'), printm 'setup object'

	fov = 250;	% 250 mm FOV

	Ndisp = 256; % display images with many pixels...
	x1d = [-Ndisp/2:Ndisp/2-1] / Ndisp * fov;
	[x1dd x2dd] = ndgrid(x1d, x1d);

	obj = mri_objects('case1');
	xtrue = obj.image(x1dd, x2dd);
	clear x1dd x2dd

	im clf, pl = inline('subplot(5, 3, 3*it+j)', 'it', 'j');
	clim = [0 2];
	if im
		pl(0,1); im(x1d, x1d, xtrue, 'x true', clim), cbar
	end
prompt
end


%
% trajectories
%

list.type = {'cartesian', 'radial', 'spiral1'}; %, 'epi-sin'};
list.arg = {{}, {}, {}}; % , {2}
list.dens = {{}, {'voronoi'}, {'voronoi'}};
N = [32 28];

% loop over trajectory types
if ~isvar('xpcg')
 for it=1:length(list.type)
	traj_type = list.type{it};

	[kspace omega wi] = mri_trajectory(traj_type, list.arg{it}, ...
		N, fov, list.dens{it});

	if im
		pl(it,1);
		plot(omega(1:5:end,1), omega(1:5:end,2), '.')
		title(sprintf('%s: %d', traj_type, size(omega,1)))
		axis(pi*[-1 1 -1 1]), axis_pipi
	end

	% create Gnufft class object
	printm 'setup G objects'
	J = [6 6];
	nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};
	mask = true(N);
	Gn = Gnufft(mask, {omega, nufft_args{:}});
	Gm = Gmri(kspace, mask, ...
		'fov', fov, 'basis', {'rect'}, 'nufft', nufft_args);

	printm 'setup data'
	ytrue = obj.kspace(kspace(:,1), kspace(:,2));

	% add noise
	randn('state', 0)
	yi = ytrue + 0 * randn(size(ytrue));

	printm 'conj. phase reconstruction'
	xcp = Gn' * (wi .* yi);
	xcp = embed(xcp, mask);

	if im
		pl(it,2); im(abs(xcp), 'Conj. Phase Recon'), cbar, drawnow
	end

	printm 'PCG with quadratic penalty'
	niter = 10;
	beta = 2^-7 * size(omega,1);	% good for quadratic
	C = Cdiff(sqrt(beta) * mask, 'edge_type', 'tight');
	xpcg = qpwls_pcg(xcp(:), Gm, 1, yi(:), 0, C, 1, niter);
	xpcg = embed(xpcg(:,end), mask);

	if im
		pl(it,3); im(abs(xpcg), '|x| pcg quad', clim), cbar, drawnow
	end
 end
end

if ~isvar('xh'), printm 'PCG with edge-preserving penalty'
	R = Robject(mask, 'edge_type', 'tight', 'type_denom', 'matlab', ...
		'potential', 'hyper3', 'beta', 2^2*beta, 'delta', 0.3);
	xh = pwls_pcg1(xpcg(:), Gm, 1, yi(:), R, 'niter', 2*niter);
	xh = embed(xh, mask);
	[magn angn] = mag_angle_real(xh);
	if im
		im(pl(it+1,1), magn, '|x| pcg edge', clim), cbar
%		im(pl(it+1,2), angn.*mask, '\angle x pcg edge', plim), cbar
	end
end
