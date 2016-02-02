% Gtomo2_wtmex_test.m
% Test the Gtomo2_wtmex object
% and the Gtomo2_dscmex object

if ~has_aspire
	return
end

if 0 % test threading timing of Gtomo2_dscmex with large case
	ig = image_geom('nx', 512, 'ny', 496, 'dx', 1);
	sg = sino_geom('par', 'nb', 600, 'na', 500, 'dr', 0.8);
	ig.mask = ig.circ(ig.fov/2) > 0;
	Gld = Gtomo2_dscmex(sg, ig, 'chat', 0);
	Gl1 = Gtomo2_dscmex(sg, ig, 'chat', 0, 'nthread', 1);

	if 1 % proj
		x = single(ig.mask);
		cpu etic
		yd = Gld * x;
		td = cpu('etoc', sprintf('%d threads', Gld.arg.nthread));
		cpu etic
		y1 = Gl1 * x;
		t1 = cpu('etoc', sprintf('%d thread', Gl1.arg.nthread));
		printm('proj speedup = %g', t1 / td)
		jf_equal(y1, yd)
	end

	if 1 % back
		y = sg.ones;
		cpu etic
		xd = Gld' * y;
		td = cpu('etoc', sprintf('%d threads', Gld.arg.nthread));
		cpu etic
		x1 = Gl1' * y;
		t1 = cpu('etoc', sprintf('%d thread', Gl1.arg.nthread));
		printm('back speedup = %g', t1 / td)
		jf_equal(x1, xd)
	end
prompt
end

if 0, printm 'test threading timing of Gtomo2_wtmex with large case'
	if ~isvar('Gwc')
		ig = image_geom('nx', 512, 'ny', 496, 'dx', 1);
		sg = sino_geom('par', 'nb', 600, 'na', 500, 'dr', 0.8);
		ig.mask = ig.circ(ig.fov/2) > 0;
		Gw1 = Gtomo2_wtmex(sg, ig, 'chat', 0, 'nthread', 1);
		Gwc = Gtomo2_wtmex(sg, ig, 'chat', 0, 'nthread', jf('ncore'));
	end

	if 1 % proj
		x = single(ig.mask);
		cpu etic
		yd = Gwc * x;
		td = cpu('etoc', sprintf('%d threads', Gwc.arg.nthread));
		cpu etic
		y1 = Gw1 * x;
		t1 = cpu('etoc', sprintf('%d thread', Gw1.arg.nthread));
		printm('proj speedup = %g', t1 / td)
		jf_equal(y1, yd)
	end

	if 1 % back
		y = sg.ones;
		cpu etic
		xd = Gwc' * y;
		td = cpu('etoc', sprintf('%d threads', Gwc.arg.nthread));
		cpu etic
		x1 = Gw1' * y;
		t1 = cpu('etoc', sprintf('%d thread', Gw1.arg.nthread));
		printm('back speedup = %g', t1 / td)
		jf_equal(x1, xd)
	end
return
prompt
end

if ~isvar('f.wtf'), printm 'make .wtf'
	ig = image_geom('nx', 22, 'ny', 20, 'dx', 2);
	sg = sino_geom('par', 'nb', 24, 'na', 18, 'dr', 1.8);
	ig.mask = ig.circ(ig.fov/2) > 0;

	f.chat = int32(0);
	f.dsc = [test_dir 't.dsc'];
	f.wtf = strrep(f.dsc, 'dsc', 'wtf');
	f.wtc = strrep(f.dsc, 'dsc', 'wtc');
	arg = aspire_pair(sg, ig, 'support', ig.mask, 'dscfile', f.dsc);
	os_run(sprintf('echo y | wt gen %s row', f.dsc)) % row grouped for OS
	os_run(sprintf('echo y | wt row2col %s %s', f.wtc, f.wtf))
end

if 1, printm 'test wtfmex asp: commands'
	nthread = int32(1);

	bufr = wtfmex('asp:read', f.wtf, f.chat);
	wtfmex('asp:print', bufr)
	tmp = wtfmex('asp:mask', bufr);
	jf_equal(tmp, ig.mask)

	mat = wtfmex('asp:load', f.wtf)';
	tmp = wtfmex('asp:load', f.wtc);
	jf_equal(mat, tmp)
	tmp = wtfmex('asp:load', f.wtc, uint8(ig.mask));
	jf_equal(mat(:,ig.mask), tmp)
	tmp = wtfmex('asp:mat', bufr, f.chat)';
	jf_equal(mat, tmp)

	x0 = single(ig.mask);
	y = wtfmex('asp:forw', bufr, nthread, x0, f.chat);
	equivs(y(:), mat * double(x0(:)));

	y0 = single(sg.ones);
	x = wtfmex('asp:back', bufr, nthread, y0, f.chat);
	equivs(x(:), mat' * double(y0(:)));

	x = wtfmex('asp:back2', bufr, nthread, y0, f.chat);
	equivs(x(:), (mat.^2)' * double(y0(:)));

	yb = wtfmex('asp:proj,block', bufr, nthread, x0, int32(1), int32(4), f.chat);
	yt = zeros(size(y), 'single'); yt(:,2:4:end) = y(:,2:4:end);
	jf_equal(yt, yb)

	xt = wtfmex('asp:back', bufr, nthread, yt, f.chat);
	xb = wtfmex('asp:back,block', bufr, nthread, yt, int32(1), int32(4), f.chat);
	jf_equal(xt, xb)

	bufc = wtfmex('asp:read', f.wtc, f.chat);
	x = wtfmex('asp:stayman2', bufc, y0);
	x = wtfmex('asp:nuyts2', bufc, y0);

%	x = wtfmex('asp:pscd', bufc, x?, dqi?, wi?, dj?); % todo

	% now the internally generated version
	tmp = aspire_pair(sg, ig, 'support', 'array');
	bufg = wtfmex('asp:gensys', tmp', 'col', uint8(ig.mask), f.chat);
	wtfmex('asp:print', bufg)
	tmp = wtfmex('asp:mask', bufg);
	jf_equal(tmp, ig.mask)

	if 1 % test writing and re-reading
		f.tmp = [test_dir 'tg.wtf'];
		fid = fopen(f.tmp, 'w');
		if (length(bufg) ~= fwrite(fid, bufg)), fail 'fwrite', end
		if (fclose(fid)), fail 'fclose', end
		tmp = wtfmex('asp:read', f.tmp, f.chat);
		jf_equal(bufg, tmp)
	end

	tmp1 = aspire_buff2mat(bufc);
	tmp2 = aspire_buff2mat(bufg);
	jf_equal(tmp1, tmp2)

	tmp = wtfmex('asp:mat', bufg, f.chat);
	jf_equal(mat, tmp)

	yb = wtfmex('asp:forw', bufg, nthread, x0, f.chat);
	equivs(y, yb)
end

if 1, printm 'test Gtomo2_wtmex'
	if 1 || ~isvar('Gwarg'), printm 'w: arg mode'
		Gwarg = Gtomo2_wtmex(arg, 'chat', f.chat);
%		G2 = Gtomo2_wtmex(arg, 'nthread', jf('ncore'), 'chat', 0);
%		tester_tomo2(G2, ig.mask) % put it through paces
%		tester_tomo2(Gwarg, ig.mask, 'G2', G2) % put it through paces
%		test_adjoint(G2);
		tester_tomo2(Gwarg, ig.mask) % put it through paces
		test_adjoint(Gwarg);
	prompt
	end

	if ~isvar('Gwtf'), printm 'w: .wtf mode'
		Gwtf = Gtomo2_wtmex(f.wtf, 'chat', 0);
		tester_tomo2(Gwtf, ig.mask) % put it through paces
		test_adjoint(Gwtf);
	prompt
	end

	if ~isvar('Gw'), printm 'w: sg,ig mode'
		Gw = Gtomo2_wtmex(sg, ig, 'chat', 0);
		tester_tomo2(Gw, ig.mask) % put it through paces
		test_adjoint(Gw);
	end
end

% todo: bug below here that makes it run slower every time i run it!?
% todo: determine if it is all modes or just some with that problem
% clear; cpu etic; Gtomo2_wtmex_test; cpu etoc
if streq(computer, 'maci')
	warn 'todo: not testing dsc version because it gets slower each time (on mac)!?'
elseif 1, printm 'test Gtomo2_dscmex'
	if ~isvar('Gdarg'), printm 'd: arg mode'
		Gdarg = Gtomo2_dscmex(arg, 'chat', 0);
		tester_tomo2(Gdarg, ig.mask) % put it through paces
		test_adjoint(Gdarg);
	end

	if ~isvar('Gdsc'), printm 'd: .dsc mode'
		Gdsc = Gtomo2_dscmex(f.dsc, 'chat', 0);
		tester_tomo2(Gdsc, ig.mask) % put it through paces
		test_adjoint(Gdsc);
	end

	if ~isvar('Gd'), printm 'd: sg,ig mode'
		Gd = Gtomo2_dscmex(sg, ig, 'chat', 0);
		tester_tomo2(Gd, ig.mask) % put it through paces
		test_adjoint(Gd);
	end
end

if 1 && isvar('Gw') && isvar('Gd'), printm 'test consistency of dsc with wtf'
	xw = Gw' * sg.ones;
	xd = Gd' * sg.ones;
	equivs(xw, xd)

	yw = Gw * ig.ones;
	yd = Gd * ig.ones;
	equivs(yw, yd)
end
