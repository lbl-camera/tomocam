 function y = jinc(x)
%function y = jinc(x)
% jinc(x) = J_1(pi x) / (2 x),
% where J_1 is Bessel function of the first kind of order 1.

x = abs(x);	% kludge for bessel with negative arguments, perhaps not needed
y = pi/4 + 0 * x; % jinc(0) = pi/4
ig = x ~= 0;
y(ig) = besselj(1, pi * x(ig)) ./ (2 * x(ig));
