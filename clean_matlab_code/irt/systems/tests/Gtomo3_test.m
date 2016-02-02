% Gtomo3_test.m
% Test the Gtomo3 object

% f3d_mex('chat', int32(2)) % for debugging

if ~isvar('G1'), printm 'setup Gtomo3_test'
	f.chat = 0;
	f.option = {};
	f.test = '3s';
	f.test = '3l';
	f.test = '2z';

	switch f.test
	case '2z'
		if 1 % small
			ig = image_geom('nx', 20, 'ny', 16, 'nz', 8, 'dx', 3.4);
			sg = sino_geom('par', 'nb', 22, 'na', 14, ...
				'dr', ig.dx, 'strip_width', 2*ig.dx);
		else
			ig = image_geom('nx', 140, 'ny', 140, 'nz', 38, 'dx', 3.4);
			sg = sino_geom('par', 'nb', 168, 'na', 192, ...
				'dr', ig.dx, 'strip_width', 2*ig.dx);
		end
		f.option = {'view2d', 1}; % needed for 2z with subsets

		tmp = Gtomo2_wtmex(sg, ig, 'mask', ig.mask_or);
		[tmp dum dum dum dum is_transpose] = ...
			wtfmex('asp:mat', tmp.arg.buff, int32(0));
		if is_transpose
			tmp = tmp'; % because row grouped
		end
		f.dir = test_dir;
		f.dsc = [test_dir 't.dsc'];
		f.wtr = strrep(f.dsc, 'dsc', 'wtr');
		delete(f.wtr)
		wtf_write(f.wtr, tmp, ig.nx, ig.ny, sg.nb, sg.na, 'row_grouped', 1)
		f.sys_type = sprintf('2z@%s@-', f.wtr);

	case '3l'
		ig = image_geom('nx', 16, 'ny', 14, 'nz', 10, ...
			'dx', 4, 'dz', 1, ...
			'offset_x', 2, 'offset_y', 1, 'offset_z', 0);
		cg = ct_geom('fan', 'ns', 80, 'nt', 40, 'na', 30, ...
			'offset_s', 0.25, ... % quarter detector
			'offset_t', 0.0, ...
			'pitch', 0, ... % test helix later
			'ds', 2, 'dt', 2, 'dso', 200, 'dod', 100, ...
			'dfs', inf); % flat detector

		if 1 % helix
			f.sys_type = aspire_pair(cg, ig, 'system', '3l');
		else
			f.sys_type = '3l@200,100,80,40,2,2,1,0,0,0,0,0@-@-2d,30@-';
		end

	case '3s'
		ig = image_geom('nx', 16, 'ny', 16, 'nz', 10, 'dx', 4, 'dz', 4);

		if 1
			f.fwhm_collimator = 1;
		 	f.fwhm_iso = 2; % depth-dependent gaussian blur
			f.psfs = '-';
			f.blur = sprintf(',gauss,%g,%g', ...
				f.fwhm_collimator, f.fwhm_iso);
		elseif 0
			f.psfs = '-';
			f.blur = ',none';
		else % stress fftw
			f.psfs = '/tmp/t,psfs.fld';
			psfs = make_3s_psfs(ny, 1, 1.2*nx, 0, 2/nx);
			f.blur = ',fft'; % fails due to fftw issues?
			fld_write(f.psfs, psfs)
		end
%		mask = []; % for 3s object!
		f.na = 6;
		f.mumap = '-';
		f.sfilter = 1;
		dx = 4;
		f.sys_type = '3s@%g,%g,%g,360,0,%d%s@%s@%s@-%d,%d,%d';
		f.sys_type = sprintf(f.sys_type, ig.dx, ig.dx, ig.dz, ...
			f.sfilter, f.blur, f.mumap, f.psfs, ig.nx, ig.nz, f.na)

%		3s@[-|sx,sy,sz,orbit,orbit_start,spline_filter[,blur_method]]
%			@mumap.file@filter.file@-nu,nv,nview
	otherwise
		fail 'bug'
	end

	f.option = {f.option{:}, 'chat', f.chat, 'checkmask', 1};

	G1 = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
		f.option{:}, 'nthread', 1);
	Gc = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
		f.option{:}, 'nthread', jf('ncore'));
	im clf, im(ig.mask, 'mask')
prompt
end

if ig.nx < 100
	tester_tomo2(G1, ig.mask, 'G2', Gc) % put it through paces
end

% todo: threads slow things down here for 2z case!?
if 0, printm 'time thread'
	cpu etic
	y1 = G1 * ig.ones;
	t1 = cpu('etoc', sprintf('%d threads', G1.arg.nthread));
	y2 = Gc * ig.ones;
	t2 = cpu('etoc', sprintf('%d threads', Gc.arg.nthread));
	printm('proj speedup = %g', t1 / t2)
        jf_equal(y1, y2)

	cpu etic
	x1 = G1' * y1;
	t1 = cpu('etoc', sprintf('%d threads', G1.arg.nthread));
	x2 = Gc' * y1;
	t2 = cpu('etoc', sprintf('%d threads', Gc.arg.nthread));
	printm('back speedup = %g', t1 / t2)
        jf_equal(x1, x2)
return
end

if 1
	x = double(convn(double(ig.mask), ones(3,3,1)/3^2, 'same') >= 1);
%	y = reshape(G * x(mask), G.arg.nn);
	ya = G1 * x;
	im clf, im pl 2 1
	im(1, ya, 'G * mask'), cbar

	% check counts scale factor (#views)
	printm('count ratio = %g', sum(ya(:)) / sum(x(:)))

	y = ones(G1.arg.nn);
%	x = embed(G' * y(:), mask);
	xa = G1' * y;
	im(2, xa, 'G''*1'), cbar
prompt
end

% todo: is below redundant with tester_tomo2?

if 1, printm 'test col vs non-col'
	yc = G1 * x(ig.mask);
	yc = reshape(yc, size(ya));
	jf_equal(yc, ya);

	xc = G1' * y(:);
	xc = ig.embed(xc);
	jf_equal(xc, xa)
end

if 1, printm 'test one vs two'
	y1 = G1 * x;
	y2 = G1 * stackup(x, x);
	jf_equal(stackup(y1,y1), y2)

	x1 = G1' * y1;
	x2 = G1' * stackup(y1,y1);
	jf_equal(stackup(x1,x1), x2)
end

% test subsets
if 1, printm 'test subsets'
	f.nblock = 3;
	Gb = Gblock(G1, f.nblock);
	yb = Gb{1} * x;
	jf_equal(ya(:,:,1:f.nblock:end), yb)

	xb = Gb{1}' * ones(size(yb));
	t = zeros(size(ya));
	t(:,:,1:f.nblock:end) = 1;
	xbf = Gb' * t;
	equivs(xb, xbf)

	im(1, yb, 'yb'), cbar
	im(2, xb, 'xb'), cbar
end
