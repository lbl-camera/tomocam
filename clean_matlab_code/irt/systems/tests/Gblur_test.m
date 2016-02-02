% Gblur_test.m
% Test the Gblur object

if ~isvar('G'),	printm 'setup Gblur_test'
	psf = [0 1 2 1 0; 1 2 4 3 1; 0 2 3 1 0];
	psf = psf / sum(psf(:));
	idim = [64 70];

	mask = true(idim);
	G = Gblur(mask, 'psf', psf, 'type', 'conv,same');
	Gf = Gblur(mask, 'psf', psf, 'type', 'fft,same');

	test_adjoint(G, 'big', 1e-11)
	test_adjoint(Gf, 'big', 1e-11)

	im clf; im pl 3 3
	im(1, G.arg.psf, 'psf'), cbar
	im(2, G.arg.mask, 'mask'), cbar
prompt
end

% test G and G'
if 1
	x = shepplogan(idim(1), idim(2), 1);
	y1 = G * x;
	y2 = Gf * x;

	x1 = G' * y1;
	x2 = Gf' * y1;

	im(3, x, 'x')
	im(4, y1, 'G * x')
	im(5, x1, 'G'' * y1')

	equivs(x1, x2)
	equivs(y1, y2)
prompt
end

if 1
	Fatrix_test_basic(G, mask) % paces
end

% check adjoint
if 1, printm 'test adjoint'
	Gs = Gblur(true(21,22), 'psf', psf, 'type', 'fft,same');
%	test_adjoint(Gs);
	test_adjoint(Gs, 'big', 1e-12)
end
