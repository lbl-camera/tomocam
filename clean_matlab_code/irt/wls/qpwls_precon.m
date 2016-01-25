 function M = qpwls_precon(type, sys, C, mask, varargin)
%function M = qpwls_precon(type, sys, C, mask, varargin)
% usage:
% M = qpwls_precon('circ0', {T}, C, mask);
% M = qpwls_precon('circ0', {G, W}, C, mask);
%
% build preconditioners for QPWLS problems
% in
%	type	string		'circ0' : circulant based on center of FOV
%				'dcd0' : diagonal / circulant / diagonal
%	sys	cell		{T} or {G, W}
%	G	[nd,np]		system matrix
%	W	[nd,nd]		data weighting matrix
%	C	[nc,np]		penalty 'derivatives' (R = \Half C'*C)
%	mask	[nx,ny]		which pixels are updated
% options
%	kappa	[nx,ny]		needed for dcd0
% out
%	M	[np,np]		Fatrix object
%
% The 'dcd0' preconditioner is based on Fessler&Booth IEEE T-IP May 1999
%
% Copyright 2004-6-29, Jeff Fessler, The University of Michigan

if nargin < 4, help(mfilename), error(mfilename), end
arg.chat = false;
arg.kappa = [];
arg = vararg_pair(arg, varargin);


%
% circulant preconditioner
%
dim = [1 1] * numel(mask);
switch type
case 'circ0'

	arg.mask = mask;
	arg.Mdft = qpwls_precon_circ_Mdft(sys, C, mask, arg.chat);
	M = Fatrix(dim, arg, 'forw', @qpwls_precon_circ_mult);

case 'dcd0'
	arg.mask = mask;
	arg = qpwls_precon_dcd0_init(sys, C, arg);
	M = Fatrix(dim, arg, 'forw', @qpwls_precon_dcd0_mult);

otherwise
	error('unknown preconditioner "%s"', type)
end


%
% qpwls_precon_dcd0_init()
% initialize dcd preconditioner based on center pixel
%
function arg = qpwls_precon_dcd0_init(sys, R, arg)
if length(sys) ~= 2, error 'need cell(2)', end
arg.diag = 1 ./ arg.kappa(arg.mask(:));

ej = zeros(size(arg.mask));
ej(end/2+1,end/2+1) = 1;
ctc = R.cgrad(R, 1e-2 * ej(arg.mask(:))) / 1e-2; % trick
ctc = embed(ctc, arg.mask);
ctc = ctc / arg.kappa(end/2+1,end/2+1)^2; % trick

sys = {sys{1}, 1}; % trick
arg.Mdft = qpwls_precon_circ_Mdft(sys, ctc, arg.mask, arg.chat);


%
% qpwls_precon_dcd0_mult()
% multiply using diag * circ * diag
%
function y = qpwls_precon_dcd0_mult(arg, x)

x = x .* arg.diag;
y = ifftn_fast(arg.Mdft .* fftn_fast(embed(x, arg.mask)));
y = y(arg.mask(:));
y = y .* arg.diag;
if isreal(x)
	y = reale(y, 'warn');
end


%
% setup circulant preconditioner based on center pixel
%
function Mdft = qpwls_precon_circ_Mdft(sys, C, mask, chat)

ej = zeros(size(mask));
iy = size(mask,2); if iy > 1, iy = iy/2+1; end
ej(end/2+1,iy) = 1;	% impulse at center
ej = ej(mask(:));

% T * x or G'WG*x
if ~iscell(sys), error 'sys must be cell', end
if length(sys) == 2
	G = sys{1};
	W = sys{2};
	gwg = G' * (W * (G * ej));
elseif length(sys) == 1
	gwg = sys{1} * ej; % T * ej
else
	error 'unknown cell'
end

gwg = embed(gwg, mask);
%im(gwg, 'gwg'), cbar, prompt

if isnumeric(C) && isequal(size(C), size(gwg)) % trick
	ccc = C;

elseif isstruct(C) % 'R'
	R = C;
	ccc = R.cgrad(R, 1e-2 * ej) / 1e-2; % trick
	ccc = embed(ccc, mask);

else
	ccc = C' * (C * ej);
	ccc = embed(ccc, mask);
end
%im(ccc, 'ccc'), cbar, prompt

f.gwg = fft2(fftshift(gwg));
f.ccc = fft2(fftshift(ccc));
f.ccc = reale(f.ccc, 'warn');		% these should be nearly real
if any(f.ccc(:) < - 1e-6 * max(f.ccc(:)))
	printm('ccc min = %g max = %g', min(f.ccc(:)), max(f.ccc(:)))
	error 'bug: circulant penalty is not nonnegative definite!?'
end
f.ccc = max(f.ccc, 0);
f.gwg = reale(f.gwg, 'warn');
if min(f.gwg(:)) < 0
	printm('setting %g%% to zero', ...
		min(f.gwg(:)) / max(f.gwg(:)) * 100)
	f.gwg = max(f.gwg, 0);
end
%minmax(f.gwg), minmax(f.ccc)
f.h = f.gwg + f.ccc;	% approximate hessian in freq. domain
if min(f.h(:)) <= 0
	error 'circulant preconditioner has zero!?  do you regularize?'
end
if chat
	printm('approximate condition number: %g', ...
		max(f.h(:)) / min(f.h(:)))
end

Mdft = 1 ./ f.h;


%
% multiply using fft
%
function y = qpwls_precon_circ_mult(arg, x)

y = ifftn_fast(arg.Mdft .* fftn_fast(embed(x, arg.mask)));
y = y(arg.mask(:));
if isreal(x)
	y = reale(y, 'warn');
end
