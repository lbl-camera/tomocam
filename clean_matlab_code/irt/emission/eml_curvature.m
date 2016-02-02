 function ni = eml_curvature(yi, ci, ri, li, yb, ctype)
%function ni = eml_curvature(yi, ci, ri, li, yb, ctype)
%
% compute surrogate parabola curvatures for Poisson emission model
% ctype:
%	oc	erdogan's optimal curvatures
%	pc	precomputed curvatures, ala empl3
%			fix: align with C version of empl2

if nargin == 1 && streq(yi, 'test'), eml_curvature_test, return, end
if nargin < 5, help(mfilename), error(mfilename), end

if ~isvar('ctype') || isempty(ctype)
	ctype = 'oc'
end

if streq(ctype, 'pc')
	if any(ci(:) < 0), error 'nonpositive ci', end
	li = (yi - ri) ./ ci;
	li = max(li, 0);
	ybi = ci .* li + ri;
	if any(ybi(:) <= 0 & yi(:) > 0), error 'model mismatch', end
%	ni = ci.^2 .* yi ./ max(ybi, eps*max([1; ybi(:)])).^2;
	% see eml_curv_pre
	ni = zeros(size(li));
	ii = li > 0;
	ni(ii) = ci(ii).^2 ./ max(yi(ii),1);
	ii = li <= 0 & ri > 0;
	ni(ii) = (ci(ii) ./ ri(ii)).^2 .* yi(ii);

elseif streq(ctype, 'oc')

	if ~isvar('yb') | isempty(yb)
		yb = ci .* li + ri;
	end

	ni_max = yi .* (ci ./ ri).^2;	% curvature at l=0
	ni = 0.001 * (ci ./ ri).^2;	% small nonzero value for yi=0

	tmp = log(yb ./ ri) - ci.*li ./ yb;
	iy = yi ~= 0;

	if 0
		il = li <= 0;
	else % trick in C program due to numerical precision issues
		il = li < 0.1 * ri ./ ci;
	end
	i = iy & il;
	ni(i) = ni_max(i);

	i = iy & ~il;
	ni(i) = 2 * yi(i) ./ li(i).^2 .* tmp(i);

	if any(ni <= 0), error 'nonpositive ni', end
	if any((ni > ni_max) & yi)
		plot([ni_max(:) ni(:) ni(:)>ni_max(:)])
		error 'large ni'
	end
%	range(ni,2)

else
	error 'unknown curvature'
end

%
% eml_curvature_test
% show figure of surrogate with optimal curvature
%
function eml_curvature_test
if 0
	l = linspace(0,2,1001)';
	l = logspace(-6,1,101)';
	n = eml_curvature(2+0*l, 1+0*l, 1+0*l, l, [], 'oc');
	semilogx(l, n, '-o')
else
	y = 3; c = 2; r = 1; l0 = 2;
	n = eml_curvature(y, c, r, l0, [], 'oc');
	l = linspace(-0.25,4,101);
	h = inline('y*log(c*l+r)-(c*l+r)', 'y', 'c', 'r', 'l');
	h0 = h(y,c,r,l0);
	derh = c * (y ./ (c*l0+r) - 1);
	n = 2/l0^2 * (h0 - h(y,c,r,0) - l0*derh);
%	nerh = c^2 * y / (c*l+r)^2;
	q = h0 + derh * (l-l0) - 0.5 * n * (l-l0).^2;
	if im
		plot(l, h(y,c,r,l), '-', l, q, 'y--', l0, h0, 'go'), grid
		xlabel 'l', legend('h(l)', 'q(l;l_0)')
		title 'Illustration of parabola surrogate for emission case'
	end
end
