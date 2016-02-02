  function ws = unwrapping_sps_manysets(w, y, delta, R, niter)
%|function ws = unwrapping_sps_manysets(w, y, delta, R, niter)
%|
%| separable quadratic surrogates phase unwrapping (w denotes frequency)
%| can be 1d, 2d, etc., depending on R.
%|
%| cost(w) = sum(i=0 to n-1) sum(j=0 to n-1)
%|		|yi*yj| (1 - cot(w*(dj-di) + \angle(yi) - \angle(yj)) + R(w)
%|
%| in
%|	w	[np 1]		initial estimate
%|	y	[np n]		n sets of measurements
%|	delta [1,n]		row vector of n offsets
%|				note - the first delta is normally just 0.
%|	R			penalty object (see Reg1.m)
%|	niter			# of iterations
%| out
%|	ws	[np niter]	iterates
%|
%| Copyright 2007-12-15, Amanda Funai, University of Michigan

if nargin < 3, help(mfilename), error(mfilename), end

if ~isvar('niter')	| isempty(niter),	niter = 1;	end
if ~isvar('chat')	| isempty(chat),	chat = logical(0);	end

if ~isvar('R') | isempty(R)
	pgrad = 0;		% unregularized default
	Rdenom = 0;
end

[np,n] = size(y);

if (isequal(n,size(delta,2))==0), help(mfilename), error(mfilename), end

ang = angle(y);

% Calculating the magnitude here to avoid recompuation each iteration.

mag = zeros(np,n*n);
wj = zeros(np,n*n);
wjtotal = zeros(np,1);

set = 1;
for i=1:n
	for j=1:n
		wj(:,set) = abs(y(:,i)).^2 .* abs(y(:,j)).^2;
		mag(:,set) = abs( conj(y(:,i)) .* y(:,j) );
		set = set+1;
	end
	wjtotal = wjtotal + abs(y(:,i)).^2;
end

for i=1:size(wj,2)
	wj(:,i) = wj(:,i) ./ wjtotal ./ abs(y(:,1)).^2;
end


%
% loop over iterations
%

ws = zeros(length(w(:)), niter);
ws(:,1) = w;

for iter = 2:niter
	if chat, printf('unwrap iteration %d', iter-1), end

	grad = 0;
	denom = 0;
	set = 1;

	% Add in num & denom contribution for each surrogate function

	for i=1:n
		for j=1:n
			s = w .* (delta(j) - delta(i)) + ang(:,i) - ang(:,j);
			grad = grad + wj(:,set) .* mag(:,set) ...
				.* (delta(j) - delta(i)) .* sin(s);
			sr = mod(s + pi, 2*pi) - pi;
			denom = denom + wj(:,set) .* mag(:,set) ...
				.* (delta(j) - delta(i))^2 .* sinc(sr / pi);
			set = set + 1;
		end
	end

	if ~isempty(R)
		pgrad = R.cgrad(R, w);
		Rdenom = R.denom(R, w);
	end

	num = grad + pgrad;
	den = denom + Rdenom;

	w = w - num ./ den;

	if chat, printf('Range %g %g', min(w), max(w)), end

	ws(:,iter) = w;
end
