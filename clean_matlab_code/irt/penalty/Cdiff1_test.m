% Cdiff1_test.m
% test Cdiff1 object

% test small size first for adjoint etc.
if 1
	ig = image_geom('nx', 8, 'ny', 6, 'dx', 1);
	for order = 0:2
		pr order
		if order == 0
			arg = {ig.dim, 'order', order, 'offset', 0};
		else
			arg = {ig.dim, 'order', order, 'offset', [1 2]};
		end

		Cc = Cdiff1(arg{:}, 'type_diff', 'convn');
		Ci = Cdiff1(arg{:}, 'type_diff', 'ind');
		Cm = Cdiff1(arg{:}, 'type_diff', 'mex');
		Cs = Cdiff1(arg{:}, 'type_diff', 'sparse');
		Cz = Cdiff1(arg{:}, 'type_diff', 'spmat');

%		Cc_f = Cc(:,:);
		Ci_f = Ci(:,:);
		Cm_f = Cm(:,:);
		Cs_f = Cs(:,:);

		if 0 % study adjoint of convn version; broken
			t2 = Cc'; t2 = t2(:,:);
			im pl 1 3
			im(1, Ci_f')
			im(2, Cc_f')
			im(3, t2)
		return
		end

%		Fatrix_test_basic(Cc, true(ig.dim))
		Fatrix_test_basic(Ci, true(ig.dim))
		Fatrix_test_basic(Cm, true(ig.dim))
		Fatrix_test_basic(Cs, true(ig.dim))
%		Fatrix_test_basic(Cz, true(ig.dim))

%		jf_equal(Ci_f, Cc_f)
		jf_equal(Ci_f, Cm_f)
		jf_equal(Ci_f, Cs_f)
		jf_equal(Ci_f, Cz)

		% abs
		Cc_a = abs(Cc); Cc_af = Cc_a(:,:);
		Ci_a = abs(Ci); Ci_af = Ci_a(:,:);
		Cm_a = abs(Cm); Cm_af = Cm_a(:,:);
		Cs_a = abs(Cs); Cs_af = Cs_a(:,:);

%		jf_equal(Cc_af, abs(Cc_f))
		jf_equal(Ci_af, abs(Ci_f))
		jf_equal(Cm_af, abs(Cm_f))
		jf_equal(Cs_af, abs(Cs_f))

		% squared
%		Cc_2 = Cc.^2; Cc_2f = Cc_2(:,:);
		Ci_2 = Ci.^2; Ci_2f = Ci_2(:,:);
		Cm_2 = Cm.^2; Cm_2f = Cm_2(:,:);
		Cs_2 = Cs.^2; Cs_2f = Cs_2(:,:);

%		jf_equal(Cc_2f, Cc_f.^2)
		jf_equal(Ci_2f, Ci_f.^2)
		jf_equal(Cm_2f, Cm_f.^2)
		jf_equal(Cs_2f, Cs_f.^2)

%		test_adjoint(Cc);
		test_adjoint(Ci);
		test_adjoint(Cm);
		test_adjoint(Cs);

%		test_adjoint(Cc_a);
		test_adjoint(Ci_a);
		test_adjoint(Cm_a);
		test_adjoint(Cs_a);

%		test_adjoint(Cc_2);
		test_adjoint(Ci_2);
		test_adjoint(Cm_2);
		test_adjoint(Cs_2);
	end
end

% timing test for large size: matlab index > sparse > mex
if 1
	ig = image_geom('nx', 2^7, 'ny', 2^7, 'nz', 2^7, 'dx', 1);
	for order = 1:2
		printm('order=%d timing tests: %d x %d', order, ig.nx, ig.ny)
		if order == 0
			arg = {ig.dim, 'order', order, 'offset', 0};
		else
			arg = {ig.dim, 'order', order, 'offset', [3 2 1]};
		end

		Cc = Cdiff1(arg{:}, 'type_diff', 'convn');
		Ci = Cdiff1(arg{:}, 'type_diff', 'ind');
		Cm = Cdiff1(arg{:}, 'type_diff', 'mex');
		Cs = Cdiff1(arg{:}, 'type_diff', 'sparse');

if 0 % quite slow!
		Cc * ig.ones; % warm up
		cpu etic
		Cc * ig.ones;
		cpu etoc 'Cdiff1 convn '
end

		Ci * ig.ones; % warm up
		cpu etic
		Ci * ig.ones;
		cpu etoc 'Cdiff1 ind   '

		Cm * ig.ones; % warm up
		cpu etic
		Cm * ig.ones;
		cpu etoc 'Cdiff1 mex   '

		cpu etic
		Cs * ig.ones;
		cpu etoc 'Cdiff1 sparse'

	end
end
