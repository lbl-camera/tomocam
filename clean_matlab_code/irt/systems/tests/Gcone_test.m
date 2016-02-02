% Gcone_test.m
% test the Gcone object
% Copyright 2008-1-1, Jeff Fessler, University of Michigan

ptypes = {'nn1', 'pd1'};
if exist('dd_ge1_mex') == 3
	ptypes{end+1} = 'dd1'; % UM only
	ptypes{end+1} = 'dd2'; % UM only
%	ptypes{end+1} = 'sp1'; % todo!
end
nn = length(ptypes);

if ~isvar('A1'), printm 'setup Gcone_test'
	f.down = 16;
	ig = image_geom('nx', 512, 'ny', 480, 'nz', 416, ...
		'dx', 1, 'dz', 0.5, ...
		'offset_x', 2.9, 'offset_y', 3.5, 'offset_z', -3.4, ...
		'down', f.down);

	cg = ct_geom('ge1', 'nt', 320, 'dt', +1.0239, ...
       		'pitch', 0, ... % test helix later
		'down', f.down);
%		'dfs', inf); % flat detector

	for ii=1:nn
		ptype = ptypes{ii};
		A1{ii} = Gcone(cg, ig, 'type', ptype, 'nthread', 1);
		Ac{ii} = Gcone(cg, ig, 'type', ptype);
	end
end


im clf, im('pl', 2, 1+nn)

if ~isvar('x0')
%	x0 = ig.circ(ig.fov/4);
%	x0 = ig.circ(ig.dx*3);
	ell = [3*ig.dx 5*ig.dx -2*ig.dz ig.dx*3 ig.dy*3 ig.zfov/4 0 0 2];
	x0 = ellipsoid_im(ig, ell, 'oversample', 2);
	im(2*(nn+1), x0)

	ya = ellipsoid_proj(cg, ell);
	im(nn+1, ya)
end

for ii=1:nn
	if 1
		printm('testing type %s', ptypes{ii})
		tester_tomo2(A1{ii}, ig.mask) % put it through paces
		tester_tomo2(Ac{ii}, ig.mask) % put it through paces
	end

	if 1 % thread checks
		y1 = A1{ii} * x0;
		yc = Ac{ii} * x0;
		im(ii, yc)
		equivs(yc, y1)
		prompt
	end

	if 1
		y0 = ya;
%		y0 = cg.ones;
%		y0 = cg.unitv;
%		y0 = cg.zeros; y0(end/2,end/2,1) = 1;
		x1 = A1{ii}' * y0;
		xc = Ac{ii}' * y0;
		im(ii+nn+1, xc)
		equivs(xc, x1)
		prompt
	end

	test_adjoint(A1{ii}, 'big', 1, 'tol', 3e-5)
	test_adjoint(Ac{ii}, 'big', 1, 'tol', 3e-5)
end

if 0
	im clf, im(ya-yc)
end
if 0
	im pl 1 3
	ia = round([1 cg.na/4 cg.na/2 cg.na]);
	im row 4
	im(1, ya(:,:,ia));
	im(2, yc(:,:,ia));
	im(3, yc(:,:,ia)-ya(:,:,ia));
	im reset
end
