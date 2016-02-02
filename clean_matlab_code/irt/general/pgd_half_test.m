 function pgd_half_test
%function pgd_half_test
% test the PGD algorithm

% cost function terms
kap = 4;
G = [1 0; 0 sqrt(kap)];
W = eye(2);
M = eye(2);
yy = 0;

niter = 19;
x = [-kap; 1];

% run PSD
xpsd = qpwls_psd(G, W, yy, x, [], M, niter);

% run PGD
data = {yy, G, W};
xpgd = pgd_half(x, data, @costgrad, M, niter);

clf, subplot(211)
plot(xpsd(1,:), xpsd(2,:), 'y-o', ...
	xpgd(1,:), xpgd(2,:), 'b--+')
xlabel x1, ylabel x2
title 'Quasi-Newton example'

%
% cost function
%
if ~isvar('qq')
	x1 = linspace(-4.5,0.5,41)';
	x2 = linspace(-1,1.5,43)';
	[xx1,xx2] = ndgrid(x1,x2);
	qq = 0 * xx1;
	for i1=1:length(x1)
		for i2=1:length(x2)
			x = [x1(i1) x2(i2)]';
			qq(i1,i2) = norm(sqrtm(W) * (yy - G * x)).^2;
		end
	end
end

if 1
	hold on
	contour(x1, x2, qq', 8)
	plot(0,0, 'rx')
	hold off
	axis equal
	axis([-4.5 0.5 -1 1.5])
	set(gca, 'xtick', -4:0)
	set(gca, 'ytick', -1:1)
	colormap(0.3+0.7*gray)
end
legend('PSD', 'PGD/Half')

if 1
	subplot(212)
	plot(1:niter, sum(abs(xpsd), 1), 'y-o', 1:niter, sum(abs(xpgd)), 'b--x')
	xlabel 'iteration', ylabel '||x||_1'
end

%
% cost function and gradient for basic WLS
%
function [cost, grad] = costgrad(x, data)
y = data{1};
G = data{2};
W = data{3};

p = G * x;
cost = 0.5 * norm(sqrtm(W) * (y - p)).^2;
if nargout > 1
	grad = -G' * (W * (y - p));
end
