 function wt = fbp_fan_short_wt(sg, varargin)
%function wt = fbp_fan_short_wt(sg, [options])
% Sinogram weighting for fan-beam short scan
% in
%	sg	strum		sino_geom
% option
%	type	'parker'	from parker:82:oss (Med Phys 1982)
% out
%	wt	[nb,na]
%
if nargin < 1, help(mfilename), error(mfilename), end
if streq(sg, 'test'), fbp_fan_short_wt_test, return, end

arg.type = 'parker';
arg = vararg_pair(arg, varargin);

switch arg.type
case 'parker'
	wt = fbp_fan_short_wt_parker(sg);
otherwise
	fail('unknown type %s', arg.type)
end


function wt = fbp_fan_short_wt_parker(sg)
nb = sg.nb;
na = sg.na;
bet = sg.ar;
gam = sg.gamma;
[g b] = ndgrid(gam, bet);
gammax = sg.gamma_max;

fun = @(x) sin(pi/2 * x).^2; % smooth out [0,1] ramp

%wt = nan(nb,na);
wt = ones(nb,na);
ii = 0 <= b & b <  2 * (gammax - g);
wt(ii) = b(ii) ./ (2 * (gammax - g(ii)));
wt(ii) = fun(wt(ii));

%ii = 2 * (gammax - g) < b & b < pi - 2 * g;
%wt(ii) = 1;

ii = pi - 2 * g < b & b <= pi + 2 * gammax;
wt(ii) = (pi + 2*gammax - b(ii)) ./ (2 * (gammax + g(ii)));
wt(ii) = fun(wt(ii));

%if any(isnan(wt(:))), warn 'nan!', end


function fbp_fan_short_wt_test
sg = sino_geom('fan', 'nb', 888, 'na', 984, 'dsd', 949, 'dod', 408, ...
	'orbit', 'short', 'down', 4);
wt = fbp_fan_short_wt(sg);
im pl 2 1
im(1, rad2deg(sg.gamma), sg.ad, wt), cbar
xlabel 'gamma [degrees]'
ylabel 'beta [degrees]'
title 'Parker weighting'
%yaxis_pi '0 p'
%plot(diff(wt,1))
%savefig fig_tomo_fan_short_wt
