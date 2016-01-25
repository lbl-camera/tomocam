 function tf = isvar(name)
%function tf = isvar(name)
% determine if "name" is a variable in the caller's workspace
% caution: using isfield(st, 'field') is faster than isvar('st.field')

if nargin < 1, help(mfilename), error(mfilename), end

tf = true;
evalin('caller', [name ';'], 'tf=false;')
