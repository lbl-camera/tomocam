%
% Gmri_gram()
% build Toeplitz-like gram-matrix object for G'WG gram matrix
% to be called by build_gram() indirectly rather than directly by user!
%
function [T, reuse] = Gmri_gram(ob, W, reuse)
if isempty(W)
	wi = 1; % default unweighted

% 1D column vector or scalar:
elseif isnumeric(W) && ndims(W) == 2 && size(W,2) == 1
	wi = W;

elseif isa(W, 'Fatrix') && streq(W.caller, 'diag_sp')
	wi = W.arg.diag;
else
	error 'Gmri_gram requires W to be diag_sp or wi array'
end
clear W

% todo:
arg2.wi = wi;
arg2.Gmri = ob;
arg2.new_zmap = @Gmri_gram_new_zmap;

T = Gmri_gram_work(arg2, reuse);


%
% Gmri_gram_work()
%
function T = Gmri_gram_work(arg2, reuse)

arg1 = arg2.Gmri.arg;
wi = arg2.wi;

if isempty(arg1.zmap)
	T = build_gram(arg1.Gnufft, wi .* abs(arg1.basis.transform).^2);
else
	LL = ncol(arg1.aB);
	arg2.T = cell(LL,1);
	reuse = [];
	for ll=1:LL
		wl = arg1.aB(:,ll) .* wi .* abs(arg1.basis.transform).^2;
		[arg2.T{ll} reuse] = build_gram(arg1.Gnufft, wl, reuse);
	end

	arg2.dim = arg1.dim([2 2]); % [np,np]
	T = Fatrix(arg2.dim, arg2, 'caller', [mfilename '.Gmri_gram'], ...
		 'forw', @Gmri_zmap_forw, ...
		 'back', @Gmri_zmap_forw); % trick: because Hermitian
end


%
% Gmri_gram_new_zmap()
% update Toeplitz-like gram-matrix Fatrix object for new zmap

function T = Gmri_gram_new_zmap(T, varargin) % (ti, zmap, L, aL)
T.arg.Gmri = T.arg.Gmri.arg.new_zmap(T.arg.Gmri, varargin{:}); % yikes!
T = Gmri_gram_work(T.arg, []);


%
% Gmri_zmap_forw(): y = T * x
%
function y = Gmri_zmap_forw(arg2, x)

arg1 = arg2.Gmri.arg;

if size(x,1) ~= arg2.dim(2)
	x = reshape(x, prod(arg1.Nd), []); % [(N),(nc)] to [*N,*nc]
	x = x(arg1.mask,:); % [np,*nc]
end
nc = ncol(x);

LL = ncol(arg1.aB);
y = 0;
for ll=1:LL
	tmp = repmat(arg1.aCt(:,ll), [1 nc]) .* x;
	tmp = arg2.T{ll} * tmp;
	tmp = repmat(conj(arg1.aCt(:,ll)), [1 nc]) .* tmp;
	y = y + tmp;
%	y = y + conj(arg1.aCt(:,ll)) .* (arg2.T{ll} * (arg1.aCt(:,ll) .* x));
end
