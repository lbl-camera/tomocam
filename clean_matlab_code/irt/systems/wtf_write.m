  function wtf_write(file, mat, nx, ny, nb, na, varargin)
%|function wtf_write(file, mat, nx, ny, nb, na, [options])
%| write (usually sparse) matrix mat to file (usually file.wtf)
%| option
%|	'row_grouped'	1|0	1 to write in row grouping (default: 0 col)
%|	'chat'
%! Copyright 2008-9-26, Jeff Fessler, University of Michigan

if nargin == 1 && streq(file, 'test'), wtf_write_test, return, end
if nargin < 6, help(mfilename), error(mfilename), end

arg.chat = 0;
arg.row_grouped = 0;
arg = vararg_pair(arg, varargin);

if ~has_mex_jf
	fail('cannot write .wtf due to mex problem')
end

jf_equal([nb*na nx*ny], size(mat)) % verify size

if ~issparse(mat)
	mat = sparse(mat);
end

wtfmex('asp:save', file, sparse(mat), int32(nx), int32(ny), ...
	int32(nb), int32(na), int32(arg.chat), int32(arg.row_grouped));


%
% wtf_write_test
%
function wtf_write_test
run_mfile_local('wtf_read test')
