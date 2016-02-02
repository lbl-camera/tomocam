%
% Reg1_setup_zxy()
%
% mex-based calculations of penalty gradient/denom for zxy ordering
%
% UNDER CONSTRUCTION!
%
function R = Reg1_setup_mex(arg, kappa)

% penalty setup
arg.beta = arg.beta(:) ...
	./ penalty_distance(arg.offsets(:), arg.dim) .^ arg.distance_power;
arg.pot_type = arg.pot_arg{1}{1};
arg.pot_params = cat(2, arg.pot_arg{1}{2:end});

mask2 = squeeze(sum(kappa, 1)) > 0; % [nz,nx,ny] to [nx,ny]
arg.cgrad_denom_arg = {...
	single(kappa), ... % assumed to be [nz,nx,ny] on input!
	int32(arg.offsets), ...
	single(arg.beta), arg.pot_type, single(arg.pot_params), ...
	uint8(mask2), int32(arg.order), int32(arg.nthread)};

if 0
% arguments that will be passed to penalty_mex() later
arg.cgrad1_str = 'cgrad,offset'; % penalty_mex() argument1
arg.cgrad1_arg = { single(kappa), int32(arg.offsets), ...
		single(arg.beta), arg.pot_type, single(arg.pot_params) ...
		int32(arg.order), int32(arg.control), int32(arg.nthread) };

arg.denom_sqs1_str = 'denom,offset'; % penalty_mex() argument1
arg.denom_sqs1_arg = { single(kappa), int32(arg.offsets), ...
		single(arg.beta), arg.pot_type, single(arg.pot_params) ...
		int32(arg.order), int32(arg.control), int32(arg.nthread) };
end

% strum methods
% trick: for backwards compatibility, all these *require* that R
% is passed (as dummy argument) even though "strum" does that.
arg.dercurv = @Reg1_com_dercurv; % trick: requires feval()
arg.cgrad_denom = @Reg1_zxy_cgrad_denom; % trick: requires feval()
meth = {...
%	'C1', @Reg1_com_C1, '(R)'; ...
%	'C', @Reg1_com_C, '(R)'; ...
%	'penal', @Reg1_com_penal, '(R, x)'; ...
%	'cgrad', @Reg1_com_cgrad, '(R, x)'; ...
	'cgrad', @Reg1_zxy_cgrad, '(R, x_zxy)'; ...
%	'egrad', @Reg1_com_egrad, '(R, x, delta)'; ...
%	'denom_sqs1', @Reg1_mex_denom_sqs1, '(R, x)'; ...
	'denom', @Reg1_zxy_denom, '(R, x_zxy)'; ...
%	'diag', @Reg1_diag, '(R)'; ...
%	'numer_pl_pcg_qs_ls', @Reg1_numer_pl_pcg_qs_ls, '(R, x1, x2)'; ...
%	'denom_pl_pcg_qs_ls', @Reg1_denom_pl_pcg_qs_ls, '(R, x1, x2)'; ...
%	'numer_denom_pl_pcg_qs_ls', @Reg1_numer_denom_pl_pcg_qs_ls, '(R, x1, x2)'
	};
R = strum(arg, meth);


%
% Reg1_zxy_cgrad_denom()
% image "x" should be in zxy order already
%
function [cgrad denom] = Reg1_zxy_cgrad_denom(sr, x)

arg = sr.cgrad_denom_arg;
[cgrad denom] = penalty_mex('cgrad,denom,zxy', single(x), arg{:});


%
% Reg1_zxy_cgrad()
% image "x" should be in zxy order already
%
function cgrad = Reg1_zxy_cgrad(sr, dummy, x)
[cgrad denom] = Reg1_zxy_cgrad_denom(sr, x);


%
% Reg1_zxy_denom()
% image "x" should be in zxy order already
%
function denom = Reg1_zxy_denom(sr, dummy, x)
[cgrad denom] = Reg1_zxy_cgrad_denom(sr, x);
