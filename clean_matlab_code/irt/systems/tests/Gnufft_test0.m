% Gnufft_test0.m
% Basic tests of small Gnufft object

% create Gnufft class object
omega = linspace(0, 10*2*pi, 101)'; % crude spiral:
omega = pi*[cos(omega) sin(omega)].*omega(:,[1 1])/max(omega);
if 1 % 2d
	N = [16 14];
	J = [8 6];
	testadj = @(sys) test_adjoint(sys);
else % 3d
	N = [16 14 10];
	J = [8 6 4];
	omega(:,3) = linspace(-pi, pi, size(omega,1));
	testadj = @(sys) test_adjoint(sys, 'big', 1);
end
K = 2*N;
mask = true(N);
mask(:,end) = false;
G = Gnufft(mask, {omega, N, J, K});

G = G.arg.new_mask(G, mask); % test mask feature

if 1
	Fatrix_test_basic(G, mask)
	testadj(G);
end

wi = [1:size(omega,1)]';
if 0
	tmp = G' * diag_sp(wi) * G;
	if length(N) == 3
		testadj(tmp);
	else
		[t0 t1] = test_adjoint(tmp);
		im(t0 - t1')
	end
return
end

if 1, printm 'Gnufft gram'
	T = build_gram(G, wi);
	Fatrix_test_basic(T, mask)
	if length(N) == 3
		testadj(T);
	else
		[t0 t1] = test_adjoint(T);
		im(t0 - t1')
	end
	warn 'todo: is there a small problem remaining in nufft_gram?'
end

if length(N) == 2, printm 'T vs G''WG' % slow in 3D
	T2 = T(:,:);
	Gf = G(:,:);
	T1 = Gf' * diag(wi) * Gf;
	max_percent_diff T1 T2
	im(T1 - T2)
%	equivs(y1, y2)
end
