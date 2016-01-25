  function [mat sp] = aspire_buff2mat(buff)
%|function [mat sp] = aspire_buff2mat(buff)
%|
%| convert 'buffer' read from .wtf to matlab sparse matrix
%| primarily for testing
%|
%| in
%|
%|	buff	byte	read using wtfmex()
%| out
%|	mat	sparse	matlab sparse matrix, full sized
%|	sp	struct	header information etc.
%|
%| Copyright 2008-9-23, Jeff Fessler, University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end
if streq(buff, 'test'), aspire_buff2mat_test, return, end

[buff sp.head] = aspire_buff_skip_ff(buff);

% methods to cast buffer bytes into appropriate types
geti = @(i, b) typecast(b((4*(min(i)-1)+1):4*(max(i)-1)+4), 'uint32');
getf = @(i, b) typecast(b((4*(min(i)-1)+1):4*(max(i)-1)+4), 'single');

% first 128 bytes is sparse header

sp.group_by = geti(1, buff);
sp.index_by = geti(2, buff);
sp.value_by = geti(3, buff);
sp.nrow_used = geti(4, buff);
sp.ncol_used = geti(5, buff);
sp.nwt = geti(6, buff);
sp.max_wt = getf(7, buff);
sp.min_wt = getf(8, buff);
sp.total = getf(9, buff);
sp.max_index = geti(10, buff);
sp.min_index = geti(11, buff);
sp.nx = geti(12, buff);
sp.ny = geti(13, buff);
sp.nb = geti(14, buff);
sp.na = geti(15, buff);
sp.fill = geti(16:32, buff);

buff = buff(129:end);

% mask
sp.mask = reshape(buff(1:(sp.nx*sp.ny)), sp.nx, sp.ny);
buff = buff((sp.nx*sp.ny+1):end);

switch sp.group_by
case 0 % by_row
	sp.ngroup = sp.nb * sp.na;
case 1 % by_col
	sp.ngroup = sp.nx * sp.ny;
otherwise
	fail('unknown group_by %d', sp.group_by)
end

% length & offset
sp.length = geti(1:sp.ngroup, buff);
buff = buff((4*sp.ngroup+1):end);
sp.offset = geti(1:sp.ngroup, buff);
buff = buff((4*sp.ngroup+1):end);

switch sp.index_by
%case 0 % by_uint2
case 1 % by_uint4
	sp.index = geti(1:sp.nwt, buff);
	buff = buff((4*sp.nwt+1):end);
otherwise
	fail('unknown index_by %d', sp.index_by)
end

switch sp.value_by
case 0 % by_float4
	sp.value = getf(1:sp.nwt, buff);
	buff = buff((4*sp.nwt+1):end);
otherwise
	fail('unknown value_by %d', sp.value_by)
end

if length(buff)
	fail 'buffer not empty at end'
end

% make sparse matrix

switch sp.group_by
case 0 % by_row
	ii = [];
	for kk=1:sp.ngroup
		ii = [ii; repmat(kk, double(sp.length(kk)), 1)];
	end
	jj = 1 + sp.index;
case 1 % by_col
	ii = 1 + sp.index;
	jj = [];
	for kk=1:sp.ngroup
		jj = [jj; repmat(kk, double(sp.length(kk)), 1)];
	end
otherwise
	fail 'bug'
end

% matlab stupidly insists on 'double' arguments!
mat = sparse(double(ii), double(jj), double(sp.value), ...
	double(sp.nb*sp.na), double(sp.nx*sp.ny), double(sp.nwt));


%
% aspire_buff_skip_ff()
%
function [buff head] = aspire_buff_skip_ff(buff)
tmp = find(buff == 12); % \f
f1 = min(tmp);
if buff(f1+1) ~= 12, error 'two form feeds', end
head = char(buff(1:(f1-1))'); % ascii header
buff = buff((f1+2):end);


%
% aspire_buff2mat_test
%
function aspire_buff2mat_test

ig = image_geom('nx', 22, 'ny', 20, 'dx', 2);
sg = sino_geom('par', 'nb', 24, 'na', 18, 'dr', 1.8);
%ig.mask = ig.circ(ig.fov/2) > 0;

sw = sg.dr * 2;
Gs = Gtomo2_strip(sg, ig, 'strip_width', sw);

pair = aspire_pair(sg, ig, 'support', 'array', 'strip_width', sw);
types = {'col', 'row'}
for it=1:length(types)
	buff = wtfmex('asp:gensys', pair', types{it}, uint8(ig.mask), int32(0));

	[mat sp] = aspire_buff2mat(buff);
	jf_equal(sp.mask, ig.mask)

	tmp = Gs.arg.G;
	equivs(mat, tmp)
end

if 0
	pr sp
	im pl 2 2
	im(1, mat)
	im(2, tmp)
	im(3, mat - tmp)
	keyboard
end
