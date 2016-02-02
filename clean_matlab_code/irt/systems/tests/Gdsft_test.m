% Gdsft_test.m
% Test the Gdsft object

%
% create Gdsft class object
%
if ~isvar('G'),	printm 'setup Gdsft_test'
	N = [32 30];
	omega = linspace(0, 10*2*pi, 201)';	% crude spiral:
	omega = pi*[cos(omega) sin(omega)].*omega(:,[1 1])/max(omega);
	pl = 330;
	if im, clf, subplot(pl+1), plot(omega(:,1), omega(:,2), '.'), end
	n_shift = N/2;
%	G = Gdsft(omega, N, 'n_shift', n_shift, 'nthread', 1);
	G = Gdsft(omega, N, 'n_shift', n_shift, 'nthread', 2);
end

%
% test data
%
if ~isvar('x'), printm 'setup data'
	x = zeros(N);
	x(5:25,10:25) = 1;
	x(15:20,15:20) = 2;
	x(15,5) = 2;
end

% compare forward
if 1
	yt = dtft(x, omega, n_shift);
	yd = G * [x(:) x(:)]; % test with two
	yd = yd(:,1);
	printf('forward max%%diff = %g', max_percent_diff(yt, yd))
end

% compare adjoint
if 1
	xt = dtft_adj(yt, omega, N, n_shift);
	xd = G' * [yt yt]; % test with two
	xd = xd(:,1);
	xd = embed(xd, G.arg.mask);

	printf('back max%%diff = %g', max_percent_diff(xt, xd))
end

if 0
	tester_tomo2(G, G.arg.mask(:));
	tester_tomo2(G2, G2.arg.mask(:));
end

if 1, printm 'test adjoint'
	Gs = Gdsft(omega, [7 8], 'n_shift', n_shift);
	test_adjoint(Gs);
end
