% denoise_threshold_test.m
% Test denoising based on l_1 type penalty, cf thresholding
% min_x |y - x|^2 + \beta * pot(x)
%
% Copyright 2005-4-22, Jeff Fessler, The University of Michigan

yi = linspace(-10,10,101)';
mask = true(size(yi));
G = diag_sp(ones(size(mask(:))));

if 1
	f.l2b_q = -1;
	f.niter = 20;
 if 1
	Rq = Robject(mask, 'type_denom', 'matlab', ...
		'offsets', 0, 'potential', 'quad', 'beta', 2^f.l2b_q);
 else
	Rq = Reg1(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'pot_arg', {'quad'}, 'beta', 2^f.l2b_q);
 end

	xq = pwls_sps_os(yi(:), yi(:), [], G, Rq, ...
			f.niter, [-inf inf], [], [], 1);
end

if 1
	f.l2b_n = 5;
 if 1
	Rc = Robject(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'potential', 'cauchy', 'delta', 0.2, 'beta', 2^f.l2b_n);
 else
	Rc = Reg1(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'pot_arg', {'cauchy', 0.2}, 'beta', 2^f.l2b_n);
 end

	xc = pwls_sps_os(yi(:), yi(:), [], G, Rc, ...
		3*f.niter, [-inf inf], [], [], 1);

 if 1
	Rh = Robject(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'potential', 'hyper3', 'delta', 0.2, 'beta', 2^f.l2b_n);
 else
	Rh = Reg1(mask, 'type_denom', 'matlab', ...
		'offsets', 0, ... % trick for identity
		'pot_arg', {'hyper3', 0.2}, 'beta', 2^f.l2b_n);
 end

	xh = pwls_sps_os(yi(:), yi(:), [], G, Rh, ...
		3*f.niter, [-inf inf], [], [], 1);
end

if im
	clf, plot(yi, yi, ':', ...
		yi, xh(:,end), '-', ...
		yi, xc(:,end), '-.', ...
		yi, xq(:,end), '--')
%	axis equal
	axis square
	legend('I', 'hyper', 'cauchy', 'quad', 2)
end
