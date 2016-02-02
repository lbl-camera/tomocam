% Gtomo2_table_test
% Test the Gtomo2_table obect by comparing with exact methods
% Jeff Fessler

% paces and adjoint
if ~isvar('Gt2'), printm 'test Gtomo2_table and its adjoint'
	sgt = sino_geom('par', 'nb', 20, 'na', 10);
	igt = image_geom('nx', 7, 'ny', 12, 'dx', 1);
	igt.mask = igt.circ > 0;
	table = {'square/strip', 'chat', 0, 'Ltab', 1000, ...
		'strip_width', sgt.dr};
	Gt1 = Gtomo2_table(sgt, igt, table, 'nthread', 1);
	Gt2 = Gtomo2_table(sgt, igt, table);
	tester_tomo2(Gt1, igt.mask, 'G2', Gt2)
	test_adjoint(Gt1);
	test_adjoint(Gt2);
%	clear sgt igt Gt1 Gt2 table
end

redo = 0;
% analytical sino
if redo | ~isvar('ya'), printm 'ya'
	down = 8;
%	down = 2; % for threading check
	ig = image_geom('nx', 512, 'ny', 496, 'fov', 500, 'down', down);
	sg = sino_geom('par', 'nb', 888, 'na', 984, ...
		'dr', 1.0, 'offset_r', 0.25, ...
		'orbit', 0*360+90, 'orbit_start', -15*0, 'down', down);

	ig.mask = ellipse_im(ig, [0 0 250*[1 1] 0 1], 'oversample', 2) > 0;
	f.mask = [test_dir 'mask.fld'];
	fld_write(f.mask, ig.mask, 'type', 'byte');

	[xa ell] = ellipse_im(ig, [], 'oversample', 2);
	im clf, im(xa, 'xa')

	ya = ellipse_sino(sg, ell, 'oversample', 8);
	im(ya, 'ya'), cbar
prompt
end


% Gtab
if redo | ~isvar('Gtab'), printm 'setup Gtab'
	if 0 % linear interpolation (cf system 9)
		f.system = 9;
		table = {'linear', 'chat', 0, 'Ltab', 1000};

	elseif 1 % square-pixel / strip-integral
		f.system = 2;

		if 0
			f.strip_width = 0;
		else
			f.strip_width = sg.dr;
		end
		f.table = {'square/strip', 'chat', 0, 'Ltab', 1000, ...
			'strip_width', f.strip_width};
	end

	cpu tic
	Gtab = Gtomo2_table(sg, ig, f.table, 'nthread', 1);
	cpu toc 'Gtab precompute time:'
	Gtab2 = Gtomo2_table(sg, ig, f.table);
prompt
end

if 0, printm 'threading check'
	cpu etic
	y1 = Gtab * xa;
	cpu etoc 'Gtab1 *:'

	cpu etic
	y2 = Gtab2 * xa;
	cpu etoc 'Gtab2 *:'
	jf_equal(y1, y2)
end

if 1 % look at DD approx
	if ~isvar('Gs')
		t = f.table; t{5} = 200;
		Gs = Gtomo2_table(sg, ig, t); % exact
		Gd = Gtomo2_table(sg, ig, {'dd2', t{2:end}});
	end
G45 = Gtomo2_table(sg, ig, {'la45', t{2:end}});
	K = Gs.arg.tab_opt.Ktab;
	L = Gs.arg.tab_opt.Ltab;
	[kk ll] = ndgrid(0:K-1, 0:L-1);
	fs = double6(Gs.arg.table);
	fd = double6(Gd.arg.table);
	f4 = double6(G45.arg.table);
	minmax(sum(fs, 1))
	minmax(sum(fd, 1))
	minmax(sum(f4, 1))
	fs = reshape(fs, L*K, []);
	fd = reshape(fd, L*K, []);
	f4 = reshape(f4, L*K, []);
	ti = col(Gs.arg.t_fun(kk, ll));
	[ti ii] = sort(ti);
	fs = fs(ii, :);
	fd = fd(ii, :);
	f4 = f4(ii, :);
	fde = fd - fs;
	f4e = f4 - fs;
	if im
		im clf, im pl 2 2
		alist = [21 32 45];
		for ii=1:3
			ia = imin(abs(sg.ad - alist(ii)));
			scale = 1;% / max(fs(:));
			im('subplot', ii)
			plot(	ti, scale * fd(:,ia)-0*fs(:,ia), 'c--', ...
				ti, scale * fs(:,ia)-0*fs(:,ia), 'r-', ...
				ti, scale * f4(:,ia)-0*fs(:,ia), 'y:')
			legend('DD', 'Exact', 'LA45')
			xlabel 'radial position'
			ylabel 'footprint'
			title(sprintf('projection view angle %g', sg.ad(ia)))
		end

%		im([1*fs, 1*fd, fde]), cbar
%		im pl 2 1,
		im subplot 4
		plot(	sg.ad, max(abs(fde), [], 1), 'c-', ...
			sg.ad, mean(abs(fde), 1), 'c-o', ...
			sg.ad, mean(abs(f4e), 1), 'y-o', ...
			sg.ad, max(abs(f4e), [], 1), 'y-')
		axis([0 180 0 1])
		legend('DD', 'DD err', 'new err', 'new')
		xtick([0:45:360])
		xlabel 'projection angle'
		ylabel 'projection error'
		title(sprintf('error for dx=%g, dr=%g', ig.dx, sg.dr))
	end
prompt
end

% dsc
if redo | ~isvar('Gdsc'), printm 'setup Gdsc'
	args = arg_pair('system', f.system, 'nx', ig.nx, 'ny', ig.ny, ...
		'nb', sg.nb, 'na', sg.na, ...
...%		'support', 'all', ...
		'support', ['file ' f.mask], ...
		'orbit', sg.orbit, 'orbit_start', sg.orbit_start, ...
		'pixel_size', ig.dx, ...
		'ray_spacing', sg.dr);
%		'flip_y', f.flip_y);
%		'source_offset', 0, 'channel_offset', -f.offset_r);

	if f.system == 9
		args = arg_pair(args, 'scale', ... % trick:
		sg.orbit / 180 / (2*pi) * (sg.dr)^2 / ig.dx, ...
			'offset_even', sg.offset_r);

	elseif f.system == 2
		args = arg_pair(args, 'scale', 0, ...
			'offset_even', sg.offset_r, ...
			'strip_width', f.strip_width);
	end

	% todo: use sg, ig
	% system object
	Gdsc = Gtomo2_dscmex(args);
prompt
end

% wtf
if redo | ~isvar('Gwtc'), printm 'setup Gwtc/Gwtr'
	if ig.nx*ig.ny <= 2^16 & sg.nb*sg.na <= 2^16
		Gwtr = Gtomo2_wtmex(args);
	end
	if ig.nx*ig.ny <= 2^16 & sg.nb*sg.na <= 2^16
		Gwtc = Gtomo2_wtmex(args, 'grouped', 'col');
	end
end


if 1, printm 'forward'
	if 0
%		x = double(ig.mask);
		x = zeros(size(ig.mask));
		x(round(ig.nx/4), round(ig.ny/3)) = 1;
	else
		x = xa;
	end
	yt = Gtab * x;
	yd = Gdsc * x;
	yc = Gwtc * x;
	yr = Gwtr * x;

	cpu etic
	yc = Gwtc * x;
	cpu etoc 'Gwtc *:'

	cpu etic
	yr = Gwtr * x;
	cpu etoc 'Gwtr *:'

	cpu etic
	yd = Gdsc * x;
	cpu etoc 'Gdsc *:'

%profile on
	cpu etic
	yt = Gtab * x;
	cpu etoc 'Gtab1 *:'
%profile report

	cpu etic
	yt2 = Gtab2 * x;
	cpu etoc 'Gtab2 *:'

	if max_percent_diff(yt, yt2), error 'thread bug', end
	max_percent_diff(yt, yd)
	max_percent_diff(yt, yr)
	max_percent_diff(yr, yc)
	im clf, im([yt; yd])
%	plot(1:na, sum(yt,1), '-', 1:na, sum(yd,1), '.')
	im(yd-yt), cbar
%	plot(1:nb, yd(:,1), '.', 1:nb, yt(:,1), 'o')
%	sum(yt(:)) ./ sum(yd(:))
%	worst = imax(abs(yt-yd),2)
%	rad2deg(ang1(worst(2)))
prompt
end


% test back projection
if 1, printm 'back'
	y = ya;
%	y = ones(size(ya));
%	y = zeros(size(ya));
%	y(round(nb/2), :) = 1;
	xd = Gdsc' * y;
	xc = Gwtc' * y;
	xr = Gwtr' * y;
	xt = Gtab' * y;
	im([xt; xr; xd])

	cpu etic
	xc = Gwtc' * y(:);
	cpu etoc 'Gwtc back:'
	xc = ig.embed(xc);

	cpu etic
	xr = Gwtr' * y(:);
	cpu etoc 'Gwtr back:'
	xr = ig.embed(xr);

	cpu etic
	xd = Gdsc' * y;
	cpu etoc 'Gdsc back:'

	cpu etic
	xt = Gtab' * y;
	cpu etoc 'Gtab1 back:'

	cpu etic
	xt2 = Gtab2' * y;
	cpu etoc 'Gtab2 back:'
	max_percent_diff(xt, xt2)

	max_percent_diff(xt, xd)
	max_percent_diff(xt, xr)
	max_percent_diff(xc, xr)
prompt
end

% hereafter requires DD
if 3 ~= exist('Gtomo_dd')
	return
end

if redo | ~isvar('Gdd'), printm 'setup Gdd'
	cpu tic
	Gdd = Gtomo_dd(sg, ig); 
	cpu toc 'Gd precompute time:'
end

if 1, printm 'tab vs dd'
	% cache warm up
	yt = Gtab * xa;
	yd = Gdd * xa;
	cpu etic
	yt = Gtab * xa;
	cpu etoc 'tab'
	cpu etic
	yd = Gdd * xa;
	cpu etoc 'dd'
	max_percent_diff(ya, yt)
	max_percent_diff(ya, yd)
	max_percent_diff(yt, yd)
return
end

return %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 0 & ~isvar('Gn'), printm 'setup Gn'
warning 'todo: update nufft'
	Gn = Gtomo_nufft(ig.mask, [nb na], ...
		'chat', 1, ...
		fan_arg{:}, ...
		'xscale', f.xscale, 'yscale', f.yscale, ...
		'orbit', f.orbit, ...
		'orbit_start', f.orbit_start, ...	
		'pixel_size', f.pixel_size, ...
		'ray_spacing', f.ray_spacing, ...
		'strip_width', f.ray_spacing, ...
		'interp', {'table', 2^11, 'minmax:kb'});
	printf('Gn precompute time = %g', toc)
end
