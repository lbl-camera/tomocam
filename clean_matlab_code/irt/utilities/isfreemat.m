 function out = isfreemat
%function out = isfreemat
% determine if this is freemat or matlab!
tmp = version;
out = strcmp(tmp, '3.5') | strcmp(tmp, '3.6');
