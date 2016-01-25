% zxy_time1.m
% timing of new zxy regularization

%	if 1 && has_mex_jf

if ~isvar('Rx'), printm 'Rx'
	ig = image_geom('nx', 512, 'ny', 496, 'nz', 520, 'fov', 500);
%	ig = image_geom('nx', 128, 'ny', 128, 'nz', 128, 'fov', 500);

%	tmp = ig.circ(ig.fov/2*1.1) > 0;
	tmp = ones(ig.dim, 'uint8');
	tmp([1 end],:,:) = 0; tmp(:, [1 end],:) = 0; % zero border for 3d
	ig.mask = tmp; clear tmp

	kappa = single(ig.mask);
	f.offsets = '3d:26';

	f.l2b = 3;
	f.pot = 'hyper3'; f.delta = 1; f.pot_arg = {f.pot, f.delta};

	order = 1;
	f.arg1 = {kappa, 'offsets', f.offsets, 'beta', 2^f.l2b, ...
		'edge_type', 'tight', 'order', order};
	f.arg = {f.arg1{:}, 'pot_arg', f.pot_arg};
	Rx = Reg1(f.arg{:}, 'type_penal', 'mex', 'control', 2);

	zxy = @(x) permute(x, [3 1 2]);
	xyz = @(x) permute(x, [2 3 1]);
	Rz = Reg1(zxy(kappa), f.arg{2:end}, 'type_penal', 'zxy');
%	Rz.offsets

	x = ig.unitv;
%	xm = x(ig.mask);
end

if 1, printm 'go'
	if 1 % check cgrad
		if order == 1 % check zxy version w/o mask
			g5 = zxy(x);
			cpu etic
			[g5 w5] = feval(Rz.cgrad_denom, Rz, g5);
			cpu etoc 'Rz cgrad/denom time'
return
			g5 = xyz(g5);
			w5 = xyz(w5);
		end
		% todo: zxy w mask

		cpu etic
		g3 = ig.embed(Rx.cgrad(Rx, xm));
		cpu etoc 'Rx cgrad time'

		equivs(g3, g5)


		if 1 % check mex version w/o mask
			cpu etic
			g4 = Rx.cgrad(Rx, x);
			cpu etoc 'Rx cgrad time'
			jf_equal(g3, g4)
		end

		if 1 % check denom
			dentf = Rt.denom_sqs1(Rt, x);
			dentm = Rt.denom_sqs1(Rt, xm);
			dentm = ig.embed(dentm);
			equivs(dentf, dentm)

			if has_mex_jf
				denxf = Rx.denom_sqs1(Rx, x);
				denxm = Rx.denom_sqs1(Rx, xm);
				denxm = ig.embed(denxm);
				equivs(dentf, denxf)
				equivs(denxm, denxf)
%				im clf, im([denxf; denxm; denxf-denxm]), cbar

				dentf = Rt.denom(Rt, x);
				denxf = Rx.denom(Rx, x);
				equivs(dentf, denxf)

				deno = ig.embed(Ro.denom(Ro, xm));
				% trick: denom matches except along outer edges
				ix = (1+order):(ig.nx-order);
				iy = (1+order):(ig.ny-order);
				equivs(dentf(ix,iy), deno(ix,iy))

				if order == 1
					equivs(dentf, w5) % check zxy
				end
			end
		end

		if 0 % check diag ?
			%d = Rm.diag(R);
		end

		% test threads
	end
prompt
end
