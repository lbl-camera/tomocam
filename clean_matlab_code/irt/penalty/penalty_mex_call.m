 function y = penalty_mex_call(diff_str, x, offsets, ndim)
%function y = penalty_mex_call(diff_str, x, offsets, ndim)
% convenience routine for calling penalty_mex
% to handle both real and complex case

if ~isreal(x)
	yr = penalty_mex(diff_str, single(real(x)), ...
		int32(offsets), int32(ndim));
	yi = penalty_mex(diff_str, single(imag(x)), ...
		int32(offsets), int32(ndim));
	y = double6(yr) + 1i * double6(yi);
else
	y = penalty_mex(diff_str, single(x), ...
		int32(offsets), int32(ndim));
	y = double6(y);
end
