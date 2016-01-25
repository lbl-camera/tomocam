 function i = imin(a, flag2d)
%function i = imin(a, flag2d)
% Return index of minimum of each column of a

if nargin == 1
	[dum, i] = min(a);
else
	i = imax(-a, flag2d);
end
		
