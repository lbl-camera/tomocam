% block_fatrix_test.m
% Test the block_fatrix object

if 1 | ~isvar('G5'), printm 'G1'
	rand('state', 0)
%	G1 = Gsparse(sparse(rand(10,20)));
	G1 = rand(10,20);
	G2 = magic(20);
	G3 = rand(10,30);
	G4 = Gnufft({'epi', [5 4], [4 4], 2*[5 4]});
	G5 = rand(size(G4));
end

Gd = block_fatrix({G1, G4}); % diag
Gc = block_fatrix({G1, G2}, 'type', 'col');
Gk = block_fatrix({G1}, 'type', 'kron', 'Mkron', 2);
Gr = block_fatrix({G1, G3}, 'type', 'row');
Gs = block_fatrix({G4, G5}, 'type', 'sum');

if 1 % test col
	x = [1:ncol(G1)]';
	y1 = G1 * x;
	y2 = G2 * x;
	yy = Gc * x;
	printm('col forw error %g', max_percent_diff([y1; y2], yy))

	x1 = G1' * y1;
	x2 = G2' * y2;
	xx = Gc' * [y1; y2];
	printm('col back error %g', max_percent_diff(x1+x2, xx))
end

if 1 % test diag
	x1 = [1:ncol(G1)]';
	x2 = [1:ncol(G4)]';
	x = [x1; x2];
	y1 = G1 * x1;
	y2 = G4 * x2;
	yy = Gd * x;
	printm('diag forw error %g', max_percent_diff([y1; y2], yy))

	x1 = G1' * y1;
	x2 = G4' * y2;
	xx = Gd' * yy;
	printm('diag back error %g', max_percent_diff([x1; x2], xx))

	Td = build_gram(Gd, [], 0);
	y1 = Td * xx;
	y2 = [G1' * G1 * x1; G4' * (G4 * x2)];
	printm('diag gram error %g%%', max_percent_diff(y1, y2))
end

if 1 % test kron
	x1 = [1:ncol(G1)]';
	x2 = [1:ncol(G1)]';
	xx = [x1; x2];
	y1 = G1 * x1;
	y2 = G1 * x2;
	yy = Gk * xx;
	printm('kron forw error %g', max_percent_diff([y1; y2], yy))

	x1 = G1' * y1;
	x2 = G1' * y2;
	xx = Gk' * yy;
	printm('kron back error %g', max_percent_diff([x1; x2], xx))
end

if 1 % test row
	x1 = [1:ncol(G1)]';
	x2 = [1:ncol(G3)]';
	y1 = G1 * x1;
	y2 = G3 * x2;
	yy = Gr * [x1; x2];
	printm('row forw error %g', max_percent_diff(y1+y2, yy))

	x1 = G1' * yy;
	x2 = G3' * yy;
	xx = Gr' * yy;
	printm('row back error %g', max_percent_diff([x1; x2], xx))
end

if 1 % test sum
	x = [1:ncol(G4)]';
	y1 = G4 * x;
	y2 = G5 * x;
	yy = Gs * x;
	printm('sum forw error %g', max_percent_diff(y1+y2, yy))
	x1 = G4' * yy;
	x2 = G5' * yy;
	xx = Gs' * yy;
	printm('sum back error %g', max_percent_diff(x1+x2, xx))
end

test_adjoint(Gc);
test_adjoint(Gd);
test_adjoint(Gk);
test_adjoint(Gr);
test_adjoint(Gs);
