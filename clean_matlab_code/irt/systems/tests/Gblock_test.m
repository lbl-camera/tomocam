% Gblock_test.m
% Test the Gblock object in all of its versions.
% This also tests several system objects:
% Gsparse, Gtomo2_dscmex, and Gtomo2_wtmex
%
% Copyright 2002-2-19, Jeff Fessler, The University of Michigan

%
% image and sparse matrix
%
if ~isvar('G0'), printm 'setup Gblock_test'
	ig = image_geom('nx', 20, 'ny', 22, 'dx', 1);
	sg = sino_geom('par', 'nb', 24, 'na', 18, ...
		'ray_spacing', 1, 'strip_width', 1);
	xt = ellipse_im(ig, 'shepplogan-emis');
	ig.mask = ig.circ > 0;

	f.strip_width = 1;
	f.strips = {'strip_width', f.strip_width};
	G1 = Gtomo2_strip(sg, ig, f.strips{:});
	G0 = G1.arg.G; % [nd,np]

	im clf, im pl 3 3
	im(1, ig.mask, 'mask')
	im(2, xt, 'xt')

	xm = xt(ig.mask);
	ym = sg.shape(G0 * xm);
	yt = G1 * xt;
	im(3, ym)
	jf_equal(yt, G1 * xt)
prompt
end


%
% 2d system objects
%
if ~isvar('Gs'), printm 'Gs'
	Gs = Gsparse(G0, 'mask', ig.mask, 'odim', [sg.nb sg.na]);
end

if has_aspire && ~isvar('Gw'), printm 'Gtomo2_wtmex'
	Gw = Gtomo2_wtmex(sg, ig, 'grouped', 'row', ...
		'pairs', {'tiny', 0, f.strips{:}});
	yy = Gw * xt;
	equivs(yy, yt)
	clear yy
prompt
end

if 1
if has_aspire && ~isvar('Gd'), printm 'Gtomo2_dscmex'
	f.dir = test_dir;
	f.mask = [f.dir 'mask.fld'];
	fld_write(f.mask, ig.mask, 'check', 0)

	if 0
		f.arg = arg_pair('system', 2, ...
			'nx', nx, 'ny', ny, 'nb', nb, 'na', na, ...
			'orbit', 180, 'orbit_start', 0, ...
			'pixel_size', 1, 'ray_spacing', 1, 'strip_width', 1, ...
			'scale', 0);
%		Gd = Gtomo2_dscmex(f.arg);
	end

if 0
	f.dsc = [f.dir 't.dsc'];
	f.com = sprintf(['wt -chat 0 dsc -support "file %s" 2' ...
			' nx %d ny %d nb %d na %d' ...
			' orbit 180 orbit_start 0' ...
			' pixel_size 1 ray_spacing 1 strip_width 1' ...
			' tiny 0 scale 0 >! %s'], ...
		f.mask, ig.nx, ig.ny, sg.nb, sg.na, f.dsc);
	os_run(f.com)

	Gd = Gtomo2_dscmex(f.dsc);
end % todo cut

	Gd = Gtomo2_dscmex(sg, ig);

	yy = Gd * xt;
	equivs(yy, yt)
	clear yy
prompt
end
else
	printm 'todo: dsc broken'
end


%
% run some basic tests/comparisons of the objects
%
if 1
	if has_aspire
		Glist = {G1, Gs, Gw, Gd};
	else
		Glist = {G1, Gs};
	end

	for ii = 1:length(Glist)
		G = Glist{ii};
		pr class(G)

		% check sum
		t1 = sum(G1);
		t2 = sum(G);
		mpd = max_percent_diff(t1,t2);
		printf('sum	mpd %g', mpd)
		if mpd/100 > 1e-6, error sum, end

		% check G*x
		t1 = G1 * xm;
		t2 = G * xm;
		mpd = max_percent_diff(t1,t2);
		printf('G*x	mpd %g', mpd)
		if mpd/100 > 1e-6, error Gx, end

		% check G*[x x]
		t1 = G1 * [xm xm];
		t2 = G * [xm xm];
		mpd = max_percent_diff(t1,t2);
		printf('G*[x x]	mpd %g', mpd)
		if mpd/100 > 1e-6, error Gxx, end

		% check G'*y
		t1 = G1' * yt;
		t2 = G' * yt;
		mpd = max_percent_diff(t1,t2);
		printf('G''y	mpd %g', mpd)
		if mpd/100 > 1e-6, error Gty, end

		% check G'*[y y]
		t1 = G1' * [yt(:) yt(:)];
		t2 = G' * [yt(:) yt(:)];
		mpd = max_percent_diff(t1,t2);
		printf('G''[y y]	mpd %g', mpd)
		if mpd/100 > 1e-6, error Gtyy, end

		% check G(:,j)
		j = find(ig.unitv);
		t1 = G1(:,j);
		t2 = G(:,j);
		mpd = max_percent_diff(t1,t2);
		printf('G(:,j)	mpd %g', mpd)
		if mpd/100 > 1e-6, error G(:,j), end

		% check G(:,js)
		j = [0 1] + j;
		t1 = G1(:,j);
		t2 = G(:,j);
		mpd = max_percent_diff(t1,t2);
		printf('G(:,js)	mpd %g', mpd)
		if mpd/100 > 1e-6, error G(:,js), end

		% check G(:,:)
		if ig.nx < 100
			t1 = G1(:,:);
			t2 = G(:,:);
			mpd = max_percent_diff(t1,t2);
			printf('G(:,:)	mpd %g', mpd)
			if mpd/100 > 1e-6, error G(:,:), end
		end

	end

prompt
end


%
% now block versions of each object
%
if ~isvar('Bs'), printm 'setup Gblock_test'

	nblock = 2;
	nblock = 8;
	Bs = Gblock(Gs, nblock, 1);
	if has_aspire
		Bd = Gblock(Gd, nblock, 1);
		Bw = Gblock(Gw, nblock, 1);
	end
prompt
end


%
% check acceleration
%
if 0
	profile on
	cpu tic
	for ii=1:10, y1 = Gs * xm; end
	cpu toc 'Gs time:'
	cpu tic
	for ii=1:10, y2 = Bs{1} * xm; end
	cpu toc 'Gb time:'
%	profile report
return
end


%
% basic block tests
%
if 1, printm 'block tests'
	if has_aspire
		Blist = {Bs, Bd, Bw};
	else
		Blist = {Bs};
	end

	for ii = 1:length(Blist)
		B = Blist{ii};

		% check mask
		try
			t1 = mask;
			t2 = B.arg.mask;
			mpd = max_percent_diff(t1,t2);
			printf('mask	mpd %g', mpd)
			if mpd/100 > 1e-6, error G(:,j), end
		catch
			% ?
		end

		% check Bx
		t1 = G1 * xm;
		t2 = B * xm;
		mpd = max_percent_diff(t1,t2);
		printf('B*x	mpd %g', mpd)
		if mpd/100 > 1e-6, error Bx, end

		% check B'y
		t1 = G1' * yt;
		t2 = B' * yt;
		mpd = max_percent_diff(t1,t2);
		printf('B''*y	mpd %g', mpd)
		if mpd/100 > 1e-6, error Bty, end

		%
		% block operations
		%
		for k=1:nblock
			ia = k:nblock:sg.na;

			% check B{k}*x
			t1 = G1 * xm;
			t1 = sg.shape(t1);
			t1 = t1(:,ia);
			t1 = t1(:);
			t2 = B{k} * xm;
			mpd = max_percent_diff(t1,t2);
			printf('B{k}*x	mpd %g', mpd)
			if mpd/100 > 1e-6, error B{k}*x, end

			% check B{k}'*y()
			tmp = sg.zeros;
			tmp(:,ia) = yt(:,ia);
			t1 = G1' * tmp(:);
			t2 = B{k}' * col(yt(:,ia));
			mpd = max_percent_diff(t1,t2);
			printf('B{k}''*y	mpd %g', mpd)
			if mpd/100 > 1e-6, error 'B{k}''*y()', end

		end
	end
prompt
end
