 function st = ct_geom(type, varargin)
%function st = ct_geom(type, varargin)
%
% Create the "CT geometry" structure that describes the sampling
% characteristics of a cone-beam CT system.
%
% in:
%	type	'fan' (multi-slice fan-beam)
%
% options for all geometries
%	'orbit_start'		default: 0
%	'orbit'			[degrees] default: 180 for parallel / mojette
%					or 360 for fan
%	'down'			down-sampling factor, for testing
%
% options for fan-beam
%	'ns'			# of horizontal samples
%	'nt'			# of vertical samples
%	'na' | 'nbeta'		# of angular samples
%	'ds'			horizontal sample spacing (default: 1)
%	'dt'			vertical sample spacing (default: -ds)
%	'offset_s'		unitless (default: 0)
%			(relative to centerline between two central channels).
%			use 0.25 or 1.25 for "quarter-detector offset"
%	'offset_t'		unitless (default: 0)
%	'zshifts' [na]		for helical: under construction
%
%	fan beam distances:
%	'dsd' | 'dis_src_det'	default: inf (parallel beam)
%	'dso' | 'dis_src_iso'	default: inf (parallel beam)
%	'dod' | 'dis_iso_det'	default: 0
%	'dfs' | 'dis_foc_src'	default: 0 (3rd generation CT arc),
%					use 'inf' for flat detector
%
% out:
%	st	(struct)	initialized structure
%
% methods:
%	st.shape(sino)		reshape sinograms that are columns into 3d array
%	st.s			s sample locations
%	st.t			t sample locations
%	st.ws			(ns-1)/2 + st.offset_s
%	st.wt			(nt-1)/2 + st.offset_t
%	st.ad			source angles in degrees
%	st.ar			source angles in radians
%	st.dim			dimensions: [st.ns st.nt st.na]
%	st.downsample(down)	reduce sampling by integer factor
%	st.ones			ones(ns,nt,na)
%	st.zeros		zeros(ns,nt,na)
%	st.rmax			max radius within FOV
%	st.shape(sino(:))	reshape to [ns,nt,na,?]
%	st.unitv(is,it,ia)	unit 'vector' with single nonzero element
%	st.plot([ig])		show geometry
%
%	trick: you can make orbit=0 and orbit_start = column vector (length na)
%	if you need nonuniformly spaced projection view angles.
%
% Copyright 2006-1-18, Jeff Fessler, The University of Michigan

if nargin == 1 && streq(type, 'test'), ct_geom_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end

if streq(type, 'ge1') % special case
	st = ct_geom_ge1(type, varargin{:});
return
end

% defaults
st.type = type;
st.ns = [];
st.nt = [];
st.na = [];
st.down = 1;
st.orbit_start = 0;

if streq(type, 'fan')
	st = ct_geom_fan(st, varargin{:});
%elseif streq(type, 'par')
%	st = ct_geom_par(st, varargin{:});
%elseif streq(type, 'moj')
%	st = ct_geom_moj(st, varargin{:});
else
	error(['unknown sinotype ' type])
end

if isempty(st.na), st.na = 2 * floor(st.ns * pi/2 / 2); end

meth = { ...
	's', @ct_geom_s, '()'; ...
	't', @ct_geom_t, '()'; ...
	'ws', @ct_geom_ws, '()'; ...
	'wt', @ct_geom_wt, '()'; ...
	'ad', @ct_geom_ad, '()'; ...
	'ar', @ct_geom_ar, '()'; ...
	'downsample', @ct_geom_downsample, '()'; ...
	'dim', @ct_geom_dim, '()'; ...
	'ones', @ct_geom_ones, '()'; ...
	'rmax', @ct_geom_rmax, '()'; ...
	'unitv', @ct_geom_unitv, '() | (is,it,ia)'; ...
	'zeros', @ct_geom_zeros, '()'; ...
	'shape', @ct_geom_shape, '()'; ...
	'plot', @ct_geom_plot, '() | (ig)';
	};

st = strum(st, meth);

if st.down ~= 1
	down = st.down; st.down = 1; % trick
	st = st.downsample(down);
end


% ct_geom_dim()
function dim = ct_geom_dim(st)
dim = [st.ns st.nt st.na];
if isempty(st.ns) || isempty(st.nt) || isempty(st.na)
	error 'dim requested without ns,nt,na'
end


% ct_geom_ones()
% sinogram of all ones
function out = ct_geom_ones(st)
out = ones(st.dim);


% ct_geom_unitv()
% sinogram with a single ray
function out = ct_geom_unitv(st, is, it, ia)
out = st.zeros;
if ~isvar('is') || isempty(is)
	is = floor(st.ns/2 + 1);
	it = floor(st.nt/2 + 1);
	ia = 1;
end
out(is,it,ia) = 1;


% ct_geom_zeros()
% sinogram of all zeros
function out = ct_geom_zeros(st)
out = zeros(st.dim);


% ct_geom_rmax()
% max radius within fov
function rmax = ct_geom_rmax(st)
smax = max(abs(st.s));
if streq(st.type, 'fan')
	if isinf(st.dso) % parallel
		rmax = smax;
	elseif st.dfs == 0 % arc
		rmax = st.dso * sin(smax / st.dsd);
	elseif isinf(st.dfs) % flat
		rmax = st.dso * sin(atan(smax / st.dsd));
	else
		error 'unknown case'
	end
end


% ct_geom_ws()
% 'middle' sample position
function ws = ct_geom_ws(st)
ws = (st.ns-1)/2 + st.offset_s;


% ct_geom_wt()
% 'middle' sample position
function wt = ct_geom_wt(st)
wt = (st.nt-1)/2 + st.offset_t;


% ct_geom_s()
% sample locations ('radial')
function s = ct_geom_s(st, varargin)
s = st.ds * ([0:st.ns-1]' - st.ws);
if length(varargin)
	s = s(varargin{:});
end

% ct_geom_t()
% sample locations ('radial')
function t = ct_geom_t(st, varargin)
t = st.dt * ([0:st.nt-1]' - st.wt);
if length(varargin)
	t = t(varargin{:});
end


% ct_geom_ad()
% angular sample locations (degrees)
function ang = ct_geom_ad(st, varargin)
ang = [0:st.na-1]'/st.na * st.orbit + st.orbit_start;
ang = ang(varargin{:});

% ct_geom_ar()
% angular sample locations (radians)
function ang = ct_geom_ar(st, varargin)
ang = deg2rad(ct_geom_ad(st));
ang = ang(varargin{:});


% ct_geom_downsample()
% down-sample (for testing)
function st = ct_geom_downsample(st, down)
st.down = st.down * down;

if ~isempty(st.zshifts)
	st.zshifts = st.zshifts(1:down:st.na);
end

st.ns = 2 * round(st.ns / down / 2); % keep it even
st.nt = 2 * round(st.nt / down / 2); % keep it even
st.na = length([1:down:st.na]);

if streq(st.type, 'fan')
	st.ds = st.ds * down;
	st.dt = st.dt * down;
else
	error(['unknown sinotype ' type])
end


% ct_geom_shape()
% reshape into sinogram array
function sino = ct_geom_shape(st, sino)
sino = reshape(sino, st.ns, st.nt, st.na, []);


%
% ct_geom_fan()
%
function st = ct_geom_fan(st, varargin);

% defaults
st.orbit = 360; % [degrees]
st.ds		= 1;
st.dt		= [];
st.offset_s	= 0;
st.offset_t	= 0;
st.zshifts	= [];

st.dsd = [];	% dis_src_det
st.dso = [];	% dis_src_iso
st.dod = [];	% dis_iso_det
st.dfs = 0;	% dis_foc_src (3rd gen CT)

subs = { ...
	'src_det_dis', 'dsd';
	'dis_src_det', 'dsd';
	'dis_src_iso', 'dso';
	'dis_iso_det', 'dod';
	'dis_foc_src', 'dfs';
	'nbeta', 'na';
	};
st = vararg_pair(st, varargin, 'subs', subs);
if isempty(st.dt), st.dt = -st.ds; end

% work out distances
if (~isempty(st.dsd) && isinf(st.dsd)) ...
|| (~isempty(st.dso) && isinf(st.dso)) % handle parallel-beam case gracefully
	st.dsd = inf; st.dso = inf; st.dod = 1;
end
if isempty(st.dsd) + isempty(st.dso) + isempty(st.dod) > 1
	error 'must provide at least two of dsd, dso, dod'
end
if isempty(st.dsd), st.dsd = st.dso + st.dod; end
if isempty(st.dso), st.dso = st.dsd - st.dod; end
if isempty(st.dod), st.dod = st.dsd - st.dso; end
if st.dso + st.dod ~= st.dsd
	error 'bad fan-beam distances'
end


%
% ct_geom_plot()
% a picture of the source position / detector geometry
%
function out = ct_geom_plot(st, ig)
if ~streq(st.type, 'fan'), error 'only fan done', end
x0 = 0;
y0 = st.dso;
t = linspace(0,2*pi,1001);
switch st.dfs
case 0
	gam = st.s / st.dsd; % 3rd gen: equiangular
case inf
	gam = atan(st.s / st.dsd); % flat
otherwise
	error 'not done'
end
xds = st.dsd * sin(gam);
yds = st.dso - st.dsd * cos(gam);
rot = deg2rad(st.orbit_start);
rot = [cos(rot) sin(rot); -sin(rot) cos(rot)];
p0 = rot * [x0; y0];
pd = rot * [xds'; yds'];
rfov = st.dso * sin(max(abs(gam)));

plot(	0, 0, '.', ...
	p0(1), p0(2), 's', ...
	[pd(1,1) p0(1) pd(1,end)], [pd(2,1) p0(2) pd(2,end)], '-', ...
	st.dso * cos(t), st.dso * sin(t), '--', ... % source circle
	rfov * cos(t), rfov * sin(t), ':', ...
	pd(1,:), pd(2,:), 'o')
if isvar('ig') && ~isempty(ig)
	hold on
	xmin = min(ig.x); xmax = max(ig.x);
	ymin = min(ig.y); ymax = max(ig.y);
	plot([xmax xmin xmin xmax xmax], [ymax ymax ymin ymin ymax], 'g-')
	hold off
end
title(sprintf('fov = %g', rfov))
axis square, zoom on
out = [];


%
% ct_geom_test()
%
function ct_geom_test
cg = ct_geom('fan', 'ns', 888, 'nt', 64, 'na', 984, ...
	'offset_s', 1.25, ...
	'dsd', 949, 'dod', 408);
cg.ad(2);
cg.downsample(2)
cg.rmax
cg.ws
cg.s(cg.ns/2+1)
cg.plot
