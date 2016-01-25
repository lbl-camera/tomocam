 function xray = xray_read_spectra(stype, varargin)
%function xray = xray_read_spectra(stype, varargin)
%
% Read X-ray spectra data and initialize a structure that describes "M"
% piecewise constant polyenergetic X-ray spectra, where M=2 for dual-kVp case.
% in
%	stype		string	which spectrum model:
%				'mono,60,100'	dual mono-energetic
%				'poly1,kvp1[,kvp2][,...]' polyenergetic
%		or:	cell	{en{1:MM}, sp{1:MM}} specify spectra
%
%	varargin	[M]	optional arguments (filtration for poly1)
%				see 'spie02' below for example.
% out
%	xray.en{1:M}	[ne]	M energy lists
%	xray.dIe{1:M}	[ne]	M differential arrays for "integrating"
%	xray.sp{1:M}	[ne]	M spectra, for M kVp settings.
%	xray.I		[M]	total intensity
%	xray.eff	[M]	effective energies
%
% Copyright 2001-04-27, Jeff Fessler, The University of Michigan

if ~nargin, help(mfilename), error(mfilename), return, end
if streq(stype, 'test'), xray_read_spectra_test, return, end


if ischar(stype)
	xray = xray_read_spectra_char(stype, varargin{:});
elseif iscell(stype) && length(stype) == 2
	xray.en = stype{1};
	xray.sp = stype{2};
	if ~iscell(xray.en) | ~iscell(xray.sp) ...
		| length(xray.en) ~= length(xray.sp)
		error 'stype as cell usage'
	end
else
	error 'bug'
end

%
% precompute some aspects of the spectra
%
for mm=1:length(xray.sp)
	xray.dIe{mm}	= xray.sp{mm} .* difff(xray.en{mm});	% I(E) dE
	xray.I(mm)	= sum(xray.dIe{mm});
	xray.eff(mm)	= sum(xray.en{mm} .* xray.dIe{mm})' ./ xray.I(mm);
	s_at_eff(mm) = interp1(xray.en{mm}, xray.sp{mm}, xray.eff(mm));
end


%
% xray_read_spectra_char()
%
function xray = xray_read_spectra_char(stype, varargin)

%
% simplest option is mono-energetic (for testing)
% usage: mono,kvp1,kvp2,...
%
if streq(stype, 'mono', 4)
	xray.kvp = str2num(strrep(stype(6:end), ',', ' '));
	for mm=1:length(xray.kvp)
		xray.en{mm} = [20:140]';
		xray.sp{mm} = xray.en{mm} == xray.kvp(mm);
	end

%
% polyenergetic spectra
% usage: poly1,kvp1,kvp2,...
%
elseif streq(stype, 'poly1', 5)
	dir_ct = ['alg' filesep 'ct'];
	dir_spectra = [path_find_dir(dir_ct) filesep 'xray-spectra'];
	if ~exist(dir_spectra, 'dir')
		error('spectra dir "%s" not in path', dir_spectra)
	end
	xray.kvp = str2num(strrep(stype(7:end), ',', ' '));

	% read raw data
	for mm=1:length(xray.kvp)
		kvp = xray.kvp(mm);
		raw = sprintf('spectra.%d', kvp);
		raw = [dir_spectra filesep raw];
		com = sprintf('tail +4 %s | head -n %d > tmp.dat', raw, kvp);
		os_run(com)
		load tmp.dat	% [ne,2]
		delete tmp.dat
		xray.en{mm} = tmp(:,1);
		xray.sp{mm} = tmp(:,2);

		% The Wilderman/Sukovic spectra must be scaled by energy!
		xray.sp{mm} = xray.sp{mm} .* xray.en{mm};

		% apply filtration, if any, using optional argument:
		% {{mtype11, thick11, mtype12, thick12, ...}, ...
		%	{mtypeM1, thickM1, mtypeM2, thickM2, ...}}
		if ~isempty(varargin)
			if length(varargin) ~= 1 | ~iscell(varargin)
				error 'filtration arguments must be cells'
			end
			xray.filts = varargin{1};
			if length(xray.filts) == 1
				filts = xray.filts{1};
			elseif length(filts) == length(xray.kvp)
				filts = xray.filts{mm};
			else
				error 'should be 1 or M sets of filters'
			end

			for ii=1:length(filts)
				filt = filts{ii};
				xray.sp{mm} = xray.sp{mm} .* ...
				xray_filter(filt{1}, filt{2}, xray.en{mm});
			end
		end
	end


%
% spectra used for 2002 SPIE talk
%
elseif streq(stype, 'spie02')
	filts = { {{'aluminum', 0.25}, {'copper', 0.05}} };
	xray = xray_read_spectra('poly1,80,140', filts);


%
% 1st spectra from predrag sukovic
%
elseif streq(stype, 'ps1')
	dir_ct = ['alg' filesep 'ct'];
	dir_spectra = [path_find_dir(dir_ct) filesep 'xray-spectra'];
	if ~exist(dir_spectra, 'dir')
		error(sprintf('spectra dir "%s" not in path', dir_spectra))
	end

	xray.kvp = [80 140];

	for mm=1:length(xray.kvp)
		file = sprintf('xray%03d.mat', xray.kvp(mm));
		file = [dir_spectra filesep file];
		if ~exist(file, 'file')
			error(sprintf('file "%s" not found', file))
		end
		raw = load(file);
		ie = raw.energy >= 20 & raw.energy <= 140;
		xray.en{mm} = raw.energy(ie);
		xray.sp{mm} = raw.spe(ie) .* raw.energy(ie);
	end


else
	error('bad stype "%s"', stype)
end


%
% test / plot routine
%
function xray_read_spectra_test

stype = 'mono,60,100';
stype = 'ps1';
stype = 'spie02';
xray = xray_read_spectra(stype);
for mm=1:length(xray.sp)
	s_at_eff(mm) = interp1(xray.en{mm}, xray.sp{mm}, xray.eff(mm));
end

clf, subplot(211)
plot(xray.en{1}, xray.sp{1}, 'y.-', xray.eff(1) * [1 1], [0 s_at_eff(1)], 'm--')
axis([10 150 0 round(1.05*max(xray.sp{1}))])
xtick([20 round(xray.eff(1)) 105 150]), ytick([0])
ylabel 'I_1(E)', title(stype)
t = sprintf('%g kVp Spectrum', xray.kvp(1));
text(80, 0.8*max(xray.sp{1}), t, 'color', 'green')

subplot(212)
plot(xray.en{2}, xray.sp{2}, 'c.-', xray.eff(2) * [1 1], [0 s_at_eff(2)], 'm--')
axis([10 150 0 round(1.05*max(xray.sp{2}))])
xtick([20 round(xray.eff(2)) 105 150]), ytick([0])
xlabel 'Energy [keV]', ylabel 'I_2(E)'
t = sprintf('%g kVp Spectrum', xray.kvp(2));
text(80, 0.8*max(xray.sp{2}), t, 'color', 'green')
