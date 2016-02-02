% cone_beam_ct_example.m
% Illustrate iterative cone-beam X-ray CT image reconstruction.
% This illustration uses a tiny system size because my cone-beam
% projector is very slow.  Users with their own fast cone-beam
% projector/backprojector can use this as a guide.
% Hopefully someday I will have faster cone-beam code...
%
% Copyright 2005-1-21, Jeff Fessler, The University of Michigan

% First run the FDK example.  It generates the true image xtrue
% and noiseless projection views "proj" and noisy data "yi"
% and generates (noisy) FDK recon "xfdk" for comparison / initialization.
if ~isvar('xfdk')
	bi = 1e6; % 1M photons / ray
	ri = 0; % no scatter etc. for now
	dfs = inf; % flat!
	feldkamp_example
prompt
end

if 0, printm 'dd check' % check distance-driven vs analytical
	Gd = Gtomo_dd(cg, ig);
	pd = Gd * xtrue;
	im clf, im_toggle(proj(:,:,1:12:end), pd(:,:,1:12:end), [0 4.4])
	nrms(pd, proj)
return
end

% 3l system matrix
if ~isvar('G'), printm 'G'
	f.sys_type = aspire_pair(cg, ig, 'system', '3l');
	G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
		'chat', 0, 'permute213', true, 'checkmask', im&0);
end

% block object for ordered-subsets iterations
if ~isvar('Gb'), printm 'Gb'
	f.nblock = 8;
	Gb = Gblock(G, f.nblock);
end

% check 3c vs analytical
if 0, printm '3l proj check'
	cpu etic
	pp = Gb * xtrue;
	cpu etoc '3l proj'
%	im clf, im_toggle(proj(:,:,1:12:end), pp(:,:,1:12:end), [0 4.4])
	nrms(pp, proj)
end

if 0 % tests
	tmp = Gb{1} * xtrue(ig.mask);
	tmp = Gb{1} * xtrue;
	ia = 1:f.nblock:cg.na;
	tmp = Gb{1}' * col(li_hat(:,:,ia));
	tmp = G * ig.unitv;
	im(tmp), cbar
	tmp = G' * tmp;
	im(tmp), cbar
return
end

% regularization object
if ~isvar('R'), printm 'regularizer'
	f.l2b = 2^4.5;
	f.delta = 100/1000;
	R = Reg1(ig.mask, 'type_denom', 'matlab', ...
                'pot_arg', {'hyper3', f.delta}, 'beta', 2^f.l2b);
	if 1 % check spatial resolution (away from edges)
		W = diag_sp(yi(:));
		psf = qpwls_psf(G, R, 1, ig.mask, W, 'fwhmtype', 'profile');
	end
end

% reshape data to be "2d arrays" for OS iterations (subset over last dim)
if ~isvar('os_data'), printm 'os_data'
	if isscalar(bi) && isscalar(ri)
		os_data = reshaper(yi, '2d');
		os_data = {os_data, ...
			bi * ones(size(os_data)), ri * ones(size(os_data))};
	else
		os_data = {reshaper(yi, '2d'), reshaper(bi, '2d'), ...
			reshaper(ri, '2d')}; % all data as 2d arrays
	end
end

% OS-SPS iterations for transmission penalized likelihood
if ~isvar('xh'), printm 'start iterations'
	f.niter = 20;
	xinit = xfdk;
	xs = tpl_os_sps(xinit(ig.mask), Gb, os_data{:}, R, 1+f.niter);
	xs = ig.embed(xs);
	xh = xs(:,:,:,end);
	im(xh)
end

% finally, compare FDK vs iterative results
if 1
	clim = [0 0.02];
	im clf
	im pl 2 2
	im(1, xtrue, 'true', clim), cbar
	im(2, xfdk, 'FDK', clim), cbar
	im(3, xh, 'PL', clim), cbar
	nrms(xtrue, xfdk)
	nrms(xtrue, xh)
end
