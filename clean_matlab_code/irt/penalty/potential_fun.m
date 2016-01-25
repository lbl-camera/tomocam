  function pot = potential_fun(type, delta, param)
%|function pot = potential_fun(type, delta, param)
%|
%| Define roughness penalty potential functions (as strum).
%|
%| The penalty will have the form
%|	R(x) = sum_k w_k * potential_k([Cx]_k, delta_k)
%| where w_k is provided elsewhere, not here!
%|
%| in
%|	type		quad, huber, hyper2, hyper3, cauchy, lange1, lange3, ...
%|			recommended: 'hyper3'
%|	delta		scalar, or image-sized array;
%|			"cutoff" parameter for edge-preserving regularization
%|	param		optional additional parameter(s) for some choices:
%|				'gf1' (generalized Fair) : [a b]
%|				'qgg2' : q
%|				'genhub' & 'stevenson94dpr' : [p q]
%|
%| out
%|	pot		strum object, with data: delta and param
%|	methods:
%|		pot.potk(C*x)	potential function value
%|		pot.wpot(C*x)	potential 'weights' (aka half-quad. curvatures)
%|		pot.dpot(C*x)	potential derivative
%|
%| trick: for now these methods all require an extra dummy argument
%| for compatability with old version as follows:
%|	pot.potk(nan, C*x)
%|
%| trick: for type 'gf1-fit', the param argument should be:
%|	{'name of potential to fit', points, param}
%| and this function returns the [a b] parameters needed for a subsequent
%| call with type 'gf1'
%|
%| Copyright 2004-5-18, Jeff Fessler, University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if streq(type, 'test') potential_fun_test, return, end

if ~isvar('delta'), delta = []; end
if ~isvar('param'), param = []; end

if streq(type, 'gf1-fit') % trick: just return parameters for this case
	pot = potential_fun_gf1_fit(param{:});
return
end

[pot.type pot.delta pot.param potk wpot dpot] = ...
	potential_fun_parse(type, delta(:), param(:));

meth = {'potk', potk, 'wpot', wpot, 'dpot', dpot};
pot = strum(pot, meth);

end % potential_fun()


%
% potential_fun_gf1_fit()
% trick: gf1-fit means match gf1 to named potential at given points [0, s1, s2]
% where the given point values are s = t / delta, i.e., in "delta" units
% param: {'name of potential to fit', points, param}
%
function ab = potential_fun_gf1_fit(type, sv, param)
sv = sv(:);
pot = potential_fun(type, 1, param); % trick: delta=1 here
pt = pot.wpot([], sv);

% ab = [1 1; -pt']' \ ((pt-1) ./ sv);

s1 = sv(1);
s2 = sv(2);
w1 = pt(1);
w2 = pt(2);
ab(1) = (w2 * s2 * (1 - w1) - w1 * s1 * (1 - w2)) ...
		/ ((w1 - w2) * s1 * s2);
ab(2) = (s2 * (1 - w1) - s1 * (1 - w2)) ...
		/ ((w1 - w2) * s1 * s2);
end % potential_fun_gf1_fit()


%
% potential_fun_parse()
%
function [type, delta, param, potk, wpot, dpot] = ...
	potential_fun_parse(type, delta, param);

dpot = [];

% trick: huber2 is just a huber with delta / 2
% so that weighting function drops to 1/2 at delta, like hyper3 etc.
if streq(type, 'huber2')
	type = 'huber';
	delta = delta / 2;
end

% trick: hyper3 is just a hyperbola with delta scaled by sqrt(3)
% to approximately "match" the properties of 'cauchy' (old erroneous 'hyper')
if streq(type, 'hyper3')
	type = 'hyper2';
	delta = delta / sqrt(3);
end

switch type

%
% quadratic potential function
%
case 'quad'
	potk = @(pot, dum, t) (abs(t).^2) / 2;
	wpot = @(pot, dum, t) ones(size(t));
	dpot = @(pot, dum, t) t;

%
% huber potential function
%
case 'huber'
	potk = @(pot, dum, t) huber_pot(t, pot.delta);
	wpot = @(pot, dum, t) huber_wpot(t, pot.delta);
	dpot = @(pot, dum, t) huber_dpot(t, pot.delta);

%
% cauchy penalty: d^2 / 2 * log(1 + (t/d)^2)
% Not convex!
%
case 'cauchy'
	potk = @(pot, dum, t) pot.delta.^2 / 2 .* log(1 + abs(t ./ pot.delta).^2);
	wpot = @(pot, dum, t) 1 ./ (1 + abs(t ./ pot.delta).^2);
	dpot = @(pot, dum, t) t ./ (1 + abs(t ./ pot.delta).^2);

%
% Geman&McClure penalty: d^2 / 2 * (t/d)^2 / (1 + (t/d)^2)
% Not convex!
%
case 'geman&mcclure'
	potk = @(pot, dum, t) pot.delta.^2 / 2 .* (t/pot.delta)^2 ./ (1 + abs(t ./ pot.delta).^2);
	wpot = @(pot, dum, t) 1 ./ (1 + abs(t ./ pot.delta).^2).^2;
	dpot = @(pot, dum, t) t ./ (1 + abs(t ./ pot.delta).^2).^2;

%
% gf1: Generalized Fair 1st-order
% wpot(t) = (1 + a * |t/d|) / (1 + b * |t/d|)
%
case 'gf1'
	potk = @(pot, dum, t) gf1_potk(t, pot.delta, pot.param(1), pot.param(2));
	wpot = @(pot, dum, t) (1 + pot.param(1) .* abs(t ./ pot.delta)) ...
		./ (1 + pot.param(2) .* abs(t ./ pot.delta));

%
% hyperbola penalty: d^2 * [ sqrt(1 + (t/d)^2) - 1 ]
%
case 'hyper2'
	potk = @(pot, dum, t) pot.delta.^2 .* (sqrt(1 + abs(t ./ pot.delta).^2) - 1);
	wpot = @(pot, dum, t) 1 ./ sqrt(1 + abs(t ./ pot.delta).^2);
	dpot = @(pot, dum, t) t ./ sqrt(1 + abs(t ./ pot.delta).^2);

case 'hyper'
	error 'use "cauchy" or "hyper3" not "hyper" now'

%
% Lange1 penalty
%
case 'lange1'
	potk = @(pot, dum, t) t.^2 / 2 ./ (1+abs(t./pot.delta));
	wpot = @(pot, dum, t) (1 + abs(t ./ pot.delta) / 2) ./ (1 + abs(t ./ pot.delta)).^2;

%
% Lange3 penalty
%
case 'lange3'
	potk = @(pot, dum, t) pot.delta.^2 .* (abs(t./pot.delta) - log(1+abs(t./pot.delta)));
	wpot = @(pot, dum, t) 1 ./ (1 + abs(t ./ pot.delta));
	dpot = @(pot, dum, t) t ./ (1 + abs(t ./ pot.delta));

%
% li98cfs
%
case 'li98cfs'
	% f = inline('atan(x) / x - 0.5'); fsolve(f, 2.3)
	delta = delta / 2.3311;
	potk = @(pot, dum, t) li98cfs_potk(t, pot.delta);
	wpot = @(pot, dum, t) li98cfs_wpot(t, pot.delta);

%
% qgg2: q-generalized gaussian for p=2, due to Thibault, Sauer, Bouman
% q = "param", same as lange1 when q=1
%
case 'qgg2'
	potk = @(pot, dum, t) t.^2 / 2 ./ (1+abs(t./pot.delta).^(2-pot.param));
	wpot = @(pot, dum, t) (1 + abs(t ./ pot.delta).^(2-pot.param) * pot.param ...
		 / 2) ./ (1 + abs(t ./ pot.delta).^(2-pot.param)).^2;

%
% genhub : generalized Huber (switch between two generalized gaussians)
% same as Huber when p=2 and q=1
%
case 'genhub'
	p = pot.param(1); % power near 0
	q = pot.param(2); % asymptotic power
	wpot = @(pot, dum, t) genhub_wpot(t, pot.delta, p, q);
	potk = @(pot, dum, t) genhub_potk(t, pot.delta, p, q);
	wpot = @(pot, dum, t) genhub_wpot(t, pot.delta, p, q);

%
% stevenson:94:dpr
% p = param(1), q = param(2), same as Huber when p=2 and q=1 ???
%
case 'stevenson94dpr'
	potk = @(pot, dum, t) stevenson94dpr_potk(t, pot.delta, pot.param(1), pot.param(2));
	wpot = @(pot, dum, t) ones(size(t)); % fix: fake for now

otherwise
	fail('Unknown potential "%s"', type)
end

if isempty(dpot) % default is t * wpot(t)
	dpot = @(pot, dum, t) t .* pot.wpot(nan, t);
end

end % potential_fun_parse()


% gf1: generalized fair 1st-order potential
function pot = gf1_potk(t, delta, a, b)
atd = abs(t ./ delta);
pot = delta.^2 ./ (2 * b.^3) * (...
	2 * b.^2 .* atd + a .* b.^2 .* atd.^2 ...
	- 2 * a .* b .* atd + 2 * (a-b) .* log(1 + b .* atd));

if 0 % symbolic check
	syms x
	syms a positive
	syms b positive
	syms t positive
	int(x*(1+a*x) / (1+b*x), x, 0, t)
end
end % gf1_potk


function pot = li98cfs_potk(t, d)
pot = d.^2 .* ((t ./ d) .* atan(t ./ d) - 0.5 * log(1 + (t ./ d).^2));
end

function pot = genhub_potk(t, d, p, q)
pot = 0.5 * abs(t) .^ p .* (abs(t) <= d) ...
 + 0.5 * (p ./ q .* d .^ (p-q) .* abs(t) .^ q ...
 + (1 - p ./ q) .* d .^ p) .* (abs(t) > d);
end

function pot = genhub_wpot(t, d, p, q)
pot = p / 2 .* d .^ (p-q) .* abs(t) .^ (q-2) .* (abs(t) > d);
ii = abs(t) <= d;
pot(ii) = p / 2 .* abs(t(ii)) .^ (p-2);
%pot = p / 2 .* abs(t) .^ (p-2) .* (abs(t) <= d) ...
% + p / 2 .* d .^ (p-q) .* abs(t) .^ (q-2) .* (abs(t) > d);
end

function pot = stevenson94dpr_potk(t, d, p, q)
pot = 0.5 * abs(t) .^ p .* (abs(t) <= d) ...
 + 0.5 * ( (p .* d .^ (p-1) .* abs(t) - p .* d .^ p ...
 + (1 ./ q) .^ (1 ./ (q-1)) ) .^ q ...
 + d .^ p - (1 ./ q) .^ (q ./ (q-1)) ) .* (abs(t) > d);
end


%
% test routine
% potential_fun_test()
% examine potential functions after rescaling.
%
function potential_fun_test

delta = 10; tmax = 0.4 * delta;
plist = {'quad', 'huber2', 'hyper3', 'lange1', 'lange3', ...
	'cauchy', 'qgg2', 'gf1-fit'};
%plist = {'quad', 'li98cfs', 'hyper3', 'huber2'}; % show li98cfs roughly hyper3
%plist = {'huber', 'genhub', 'quad'};%, 'stevenson94dpr'};
%plist = {'genhub'}
%plist = {'hyper3', 'qgg2'}; delta = 20; tmax = 10;
%plist = {'qgg2', 'gf1-fit'};
%plist = {'quad'};
t = tmax * linspace(-delta, delta, 401)';
for ii=1:length(plist)
	type = plist{ii};
	if streq(type, 'quad')
		leg{ii} = type;
	else
		leg{ii} = [type ' \delta = ' num2str(delta)];
	end

	switch type
	case 'gf1-fit'
		param = potential_fun('gf1-fit', nan, {'qgg2', [1 10], 1.2});
		type = 'gf1'; % trick
		leg{ii} = [leg{ii} sprintf(' %.3g %.4f', param(1), param(2))];
	case 'qgg2'
		param = 1.2;
		leg{ii} = [leg{ii} ' q = ' num2str(param)];
	case 'genhub'
		param = [2.0 1.1];
		leg{ii} = [leg{ii} sprintf('p=%g q=%g', param(1), param(2))];
	case 'stevenson94dpr'
		param = [2 2.01];
		leg{ii} = [leg{ii} sprintf('p=%g q=%g', param(1), param(2))];
	otherwise
		param = [];
	end

	pot = potential_fun(type, delta, param);
	pp(:,ii) = pot.potk(nan, t);
	pw(:,ii) = pot.wpot(nan, t);
	pd(:,ii) = pot.dpot(nan, t);

	if 0 % test vs old
		opot = potential_func(type, delta, param);
		opp = opot.potk(opot, t);
		opw = opot.wpot(opot, t);
		opd = opot.dpot(opot, t);
		if any(opp ~= pp(:,ii)), 'p bug', type, keyboard, end
		if any(opw ~= pw(:,ii)), 'w bug', type, keyboard, end
		if any(opd ~= pd(:,ii)), 'd bug', type, keyboard, end
	end
end

if im
	clf
	subplot(411), plot(t, pp), title 'potk'
	axis tight, axisy([-0.0 2.5] * delta^2)
	legend(leg, 2)
	subplot(412), plot(t, pw), title 'wpot'
	axis tight, axisy(0, 1.1)
	subplot(413), plot(t, pd), title 'dpot'
	axis tight, axisy([-1 1] * 1.1 * delta)
	% check derivatives
	subplot(414)
	plot(t, pd)
	title 'dpot check'
	hold on
	d = diffc(pp) / (t(2)-t(1));
	plot(t(1:end-1), d(1:end-1,:), '--')
	hold off
	axis tight, axisy([-1 1] * 1.1 * delta)
end

end % potential_fun_test()
