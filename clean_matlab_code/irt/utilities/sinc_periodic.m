 function x = sinc_periodic(t, K)
%function x = sinc_periodic(t, K)
% periodic sinc function, obtained by replicates of a sinc:
% x(t) = \sum_l sinc(t - l K) = ... = sin(pi*t) / tan(pi*t/K) / K for K even.
% This function is bandlimited and its samples are an impulse train.
% It is closely related to the Dirichlet function diric() for odd K,
% but it differs for even K.
% Copyright 2003-11-2, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if streq(t, 'test'), sinc_periodic_test, return, end

if ~rem(K,2)	% even
	d = tan(pi*t/K);
	j = abs(d) > 1e-12;
	x = ones(size(t));
	t = t(j);
	d = d(j);
	x(j) = sin(pi*t) ./ d / K;
else
	x = diric(2*pi*t/K,K);
end


%
% self test
%
function sinc_periodic_test

Klist = [4 5];
im clf, pl=220;
for kk=1:2
	K = Klist(kk);
	n = [0:(4*K)]';
	t = linspace(0,4*K,401)';
	x = inline('sinc_periodic(t, K)', 't', 'K');
	y = inline('diric(2*pi*t/K,K)', 't', 'K');
	if im
		subplot(pl+kk+0)
		plot(t, x(t,K), '-', n, x(n,K), 'o'), title 'Sinc-Periodic'
		axis([0 4*K -1 1]), xtick([0:4]*K), grid
		subplot(pl+kk+2)
		plot(t, y(t,K), '-', n, y(n,K), 'o'), title 'Dirichlet'
		axis([0 4*K -1 1]), xtick([0:4]*K), grid
		ylabelf('K=%d', K)
	end
end
