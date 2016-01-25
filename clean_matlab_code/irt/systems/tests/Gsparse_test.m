% Gsparse_test.m
% test the Gsparse object

if 1
	ig = image_geom('nx', 20, 'ny', 18, 'dx', 1)
	nx = ig.nx;
	ny = ig.ny;
	ig.mask = ig.circ > 0;
	nb = 21; na = 9;
	Gl = Glinear(ig.nx, ig.ny, nb, na, 1, ig.mask, 0);
	im clf, im pl 3 3; im(1, ig.mask, 'mask')
	if im, im subplot 2, spy(Gl), title 'Gl', end
end


if 1, printm 'test ordinary sparse matrix'
	G1 = Gsparse(Gl, 'mask', ig.mask, 'odim', [nb na]);

	x = double6(ig.mask);
	y1 = G1 * x;
	im(4, y1, 'G*x')

	x1 = G1' * y1;
	im(7, x1, 'G''y')

	Fatrix_test_basic(G1, ig.mask);
	test_adjoint(G1);
end


if 1, printm 'test multiple rhs'
	j = find(G1.arg.mask);
	j = find(j == sub2ind([nx ny], nx/2, ny/2+2));
	t = reshape(G1(:,j), nb, na);
	im(3, t, 'G(:,j)'), cbar

	t = reshape(G1(:,[j j+1]), [nb na 2]);
	im(6, t, 'G(:,[j j+1])')

	i = sub2ind([nb na], nb/2, na/2);
	t = embed(G1([i i+2],:)', ig.mask);
	im(9, t, 'G([i i+1],:)')
end


if 1, printm 'test blocks'
	nblock = 4;
	Gb = Gblock(G1, nblock, 1);
	x = double6(ig.mask);
	istart = 3;
	ia = istart:nblock:na;
	y1 = Gb{istart} * x;
	y2 = Gb * x;
	if any(col(y1-y2(:,ia))), error 'bug', end

	x1 = Gb{istart}' * y1;
	y2 = zeros(size(y2));
	y2(:,ia) = y1;
	x2 = Gb' * y2;
	if any(abs(col(x1-x2)) > 2e-6), error 'bug', end
end


if 1, printm 'test power'
	y1 = G1.^2 * x(ig.mask);
	y2 = Gl(:,ig.mask).^2 * double(x(ig.mask));
	if any(abs(col(y1-y2)) > 2e-6), error 'bug', end
end

if 1, printm 'test cell args'
	[i j s] = find(Gl);
	s = dsingle(s);
	G2 = Gsparse({i,j,s,nb*na,nx*ny}, 'mask', ig.mask, 'odim', [nb na]);

	x = double6(ig.mask);
	y1 = G2 * x;
	y2 = reshape(G2 * x(ig.mask), [nb na]);
	im(5, y1, 'G*x')
	if any(col(y1-y2)), error 'bug', end

	x1 = G2' * y1;
	x2 = embed(G2' * y1(:), ig.mask);
	im(8, x1, 'G''y')
	if any(col(x1-x2)), error 'bug', end

	test_adjoint(G2);
end


if ~has_aspire
	printm 'skipping since no aspire'
	return
end

if 1, printm 'test file.wtf'
	f.dir = test_dir;
	f.wtf = [test_dir 't.wtf'];
	delete(f.wtf)
	wtf_write(f.wtf, Gl, nx, ny, nb, na)

	G3 = Gsparse(f.wtf, 'mask', ig.mask);

	x = double6(ig.mask);
	y1 = G3 * x;
	y2 = reshape(G3 * x(ig.mask), [nb na]);
	im(6, y1, 'G*x')
	if any(col(y1-y2)), error 'bug', end

	x1 = G3' * y1;
	x2 = embed(G3' * y1(:), ig.mask);
	im(9, x1, 'G''y')
	if any(col(x1-x2)), error 'bug', end

	test_adjoint(G3);
end
